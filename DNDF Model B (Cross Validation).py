import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, accuracy_score,
    recall_score, precision_score, f1_score, confusion_matrix
)
import random

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare data
data = pd.read_csv("Data/revised_clinical_with_new_methylation_data.csv")

# Separate features and labels
X = data.drop(columns=["target"]).values
y = data["target"].values

# ==========================
# Class definitions
# ==========================
class NeuralFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(NeuralFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return x

class DNDFTree(nn.Module):
    def __init__(self, depth, input_dim, num_classes, feature_subset=None):
        super(DNDFTree, self).__init__()
        if feature_subset is None:
            self.feature_subset = torch.arange(input_dim)
        else:
            self.feature_subset = feature_subset
        self.depth = depth
        self.num_classes = num_classes
        self.num_leaf_nodes = 2 ** depth

        self.decision_layer = nn.Linear(len(self.feature_subset), self.num_leaf_nodes)
        self.leaf_distributions = nn.Parameter(torch.rand(self.num_leaf_nodes, num_classes))

    def forward(self, x):
        x = x[:, self.feature_subset]

        batch_size = x.size(0)
        decision_logits = self.decision_layer(x)
        decision_probs = torch.sigmoid(decision_logits)

        mu = torch.ones(batch_size, 1, device=x.device)
        for d in range(self.depth):
            indices = torch.arange(2 ** d).to(x.device)
            probs = decision_probs[:, indices]
            mu = mu.unsqueeze(2)
            mu = torch.cat([mu * probs.unsqueeze(2), mu * (1 - probs).unsqueeze(2)], dim=2)
            mu = mu.view(batch_size, -1)
        mu = mu.view(batch_size, self.num_leaf_nodes)

        leaf_distributions = F.softmax(self.leaf_distributions, dim=-1)
        output = torch.matmul(mu, leaf_distributions)
        return output

class DNDF(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_trees=5, tree_depth=3):
        super(DNDF, self).__init__()
        self.num_classes = num_classes
        self.feature_extractor = NeuralFeatureExtractor(input_dim, hidden_dim)

        feature_indices = np.arange(hidden_dim)
        self.trees = nn.ModuleList([
            DNDFTree(
                tree_depth,
                len(feature_subset := torch.tensor(
                    np.random.choice(feature_indices, size=int(0.7 * hidden_dim), replace=False)
                )),
                num_classes,
                feature_subset=feature_subset
            )
            for _ in range(num_trees)
        ])

    def forward(self, x):
        x = self.feature_extractor(x)
        tree_outputs = [tree(x) for tree in self.trees]
        forest_output = torch.mean(torch.stack(tree_outputs), dim=0)
        return forest_output

# ==========================
# Define training and evaluation functions outside of the class
# ==========================
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs[:, 1].cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    auprc = auc(recall, precision)
    auroc = roc_auc_score(y_true, y_probs)
    accuracy = accuracy_score(y_true, y_pred)
    recall_score_value = recall_score(y_true, y_pred)
    precision_score_value = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"AUPRC: {auprc:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision_score_value:.4f}")
    print(f"Recall: {recall_score_value:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return auprc, auroc, accuracy, recall_score_value, precision_score_value, f1

# ==========================
# Cross-validation + Repetitions
# ==========================
n_splits = 5

# Arrays to store overall metrics across all 30 runs
results_list = []

for i in range(30):   
    seed = i
    print(f"\n=== Run {i+1}/30 with seed {seed} ===")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Arrays to store performance metrics for each 5-fold cycle (one run)
    auprc_scores = []
    auroc_scores = []
    accuracy_scores = []
    recall_scores = []
    precision_scores = []
    f1_scores = []

    # Perform the 5-fold cross-validation
    fold = 0
    for train_index, test_index in skf.split(X, y):
        fold += 1
        print(f"\nFold {fold}")
        # Split the data for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Standardize features based on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create DataLoaders
        train_ds = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
        test_ds = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)
        
        # Compute class weights
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        # Model parameters
        input_dim = X_train.shape[1]
        hidden_dim = 64
        num_classes = len(np.unique(y_train))
        num_trees = 5
        tree_depth = 3
        
        # Initialize the model
        model = DNDF(input_dim, hidden_dim, num_classes, num_trees, tree_depth).to(device)
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        train_model(model, train_ds, criterion, optimizer, device, epochs=50)
        
        # Evaluate the model
        auprc, auroc, accuracy, recall_value, precision_value, f1 = evaluate_model(model, test_ds, device)
        
        # Store the fold metrics
        results_list.append({
            'run': i+1,
            'fold': fold,
            'AUPRC': auprc,
            'AUROC': auroc,
            'Accuracy': accuracy,
            'Precision': precision_value,
            'Recall': recall_value,
            'F1_Score': f1
        })

# After all runs and folds have been recorded
results_df = pd.DataFrame(results_list)
results_df.to_csv('all_runs_all_folds_results.csv', index=False)

# Compute the mean metrics across all runs and folds
mean_metrics = results_df[['AUPRC', 'AUROC', 'Accuracy', 'Precision', 'Recall', 'F1_Score']].mean()

# Convert to a DataFrame for easy saving
mean_metrics_df = pd.DataFrame(mean_metrics, columns=['Mean'])
mean_metrics_df.to_csv('average_metrics_after_30_runs.csv', index=True)
