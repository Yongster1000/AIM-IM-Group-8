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
import joblib
import csv

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
def fix_seed(seed: int):
    # Set the random seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    # If you're using GPU (CUDA), you may want to set these as well
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you have more than one GPU

    # Ensuring deterministic behavior for CUDA operations
    torch.backends.cudnn.deterministic = True
fix_seed(42)
categorical_features = ['SEX', 'edu', 'DM_FAM', 'smoking', 'DRK', 'betel', 'SPORT', 'cardio_b']
continuous_features = ['AGE', 'SBP', 'DBP', 'HR', 'Weight', 'Height', 'BMI', 'WHR', 'T_CHO', 'TG', 'HDL', 'LDL']
csv_file = 'Data/methylation_filtered_final_new.csv'
with open(csv_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    list = [row for row in spamreader]
    #gene_features = list[0][0].split(',')[1:-1]

#attr_list = categorical_features + continuous_features + gene_features
attr_list = categorical_features + continuous_features
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load and prepare data
data = pd.read_csv("Data/revised_clinical_data.csv")
print("before:")
print(data)
data = data.drop(columns=data.columns[0])  # Drop index column
print("after:")
print(data)

class FocalLoss(nn.Module):
  def __init__(self, weight=None, gamma=2., reduction='none'):
    nn.Module.__init__(self)
    self.weight = weight
    self.gamma = gamma
    self.reduction = reduction
      
  def forward(self, input_tensor, target_tensor):
    log_prob = F.log_softmax(input_tensor, dim=-1)
    prob = torch.exp(log_prob)
    return F.nll_loss(
        ((1 - prob) ** self.gamma) * log_prob, 
        target_tensor, 
        weight=self.weight,
        reduction = self.reduction
    )

def preprocess_data(X):
    
    # Handle outliers by clipping to the 1st and 99th percentiles
    lower_bound = np.percentile(X, 1, axis=0)
    upper_bound = np.percentile(X, 99, axis=0)
    
    X = np.clip(X, lower_bound, upper_bound)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

# Separate features and labels
X = data.drop(columns=["target"])#.values
X = X[attr_list]
y = data["target"]#.values
print(X.shape, y.shape)

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
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs[:, 1].cpu().numpy())

    # Convert results to arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Compute metrics
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    auprc = auc(recall, precision)
    auroc = roc_auc_score(y_true, y_probs)
    accuracy = accuracy_score(y_true, y_pred)
    recall_value = recall_score(y_true, y_pred, zero_division=0)
    precision_value = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"AUPRC: {auprc:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision_value:.4f}")
    print(f"Recall: {recall_value:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return auprc, auroc, accuracy, recall_value, precision_value, f1
# ==========================
# Cross-validation + Repetitions
# ==========================
n_splits = 5
batch_size = 32


# Arrays to store overall metrics across all 30 runs
results_list = []

for i in range(30):   
    seed = i
    print(f"\n=== Run {i+1}/30 with seed {seed} ===")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):

        print(f"\nFold {fold}")

        # Split the data
        #X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = X.copy().iloc[train_index], X.copy().iloc[test_index]
        y_train, y_test = y.copy().iloc[train_index], y.copy().iloc[test_index]
        include_methylation = False     # change if methylation
        if include_methylation:
            continuous_features.extend(gene_features)
        train_df = X_train
        test_df = X_test
        for feature in continuous_features:
            z_scores = np.abs((train_df[feature] - train_df[feature].mean()) / train_df[feature].std())
            z_score_filtered = train_df[z_scores < 3]
        #### Interquartile Range (IQR) Method
    
        feature_data = train_df.dropna()
        for feature in continuous_features:
            Q1, Q3 = np.percentile(feature_data[feature] , [25, 75])
            IQR = Q3 - Q1
            IQR_filtered_data = feature_data[(feature_data[feature]  >= Q1 - 1.5 * IQR) & (feature_data[feature]  <= Q3 + 1.5 * IQR)]
        #### Median Absolute Deviation (MAD)
        '''MAD is a robust method based on the median and measures the absolute deviation of each value from the median. 
        Dividing by MAD (or scaled by 1.482 for a normal approximation) helps identify outliers by setting a threshold (e.g., 3 MAD).
        MAD is more robust than Z-score for non-normal distributions and is less influenced by extreme values.
        '''
        feature_data = train_df.dropna()
        for feature in continuous_features:
            median = np.median(feature_data[feature])
            mad = np.median(np.abs(feature_data[feature] - median))
            MAD_filtered_data = feature_data[np.abs(feature_data[feature] - median) / mad < 3]
    
        overlapping_indices = z_score_filtered.index.intersection(MAD_filtered_data.index).intersection(IQR_filtered_data.index)
        filtered_train_df = z_score_filtered.loc[overlapping_indices]

        ### Imputation
        fillna = {}
        train_df_columns = filtered_train_df.columns.to_list()

        for i in range(0,len(train_df_columns)):
            if train_df_columns[i] in categorical_features:
                fillna[train_df_columns[i]] = filtered_train_df[train_df_columns[i]].mode().values[0]
            if train_df_columns[i] in continuous_features:
                fillna[train_df_columns[i]] = filtered_train_df[train_df_columns[i]].mean()
        #print('fillna =', fillna)
        for index, row in train_df.iterrows():
            for j in train_df:
                if pd.isna(row[j]):
                    #print('index, j =', index, j)
                    #print(train_df.loc[index, j])
                    train_df.loc[index, j] = fillna[j]
        for index, row in test_df.iterrows():
            for j in test_df:
                if pd.isna(row[j]):
                    test_df.loc[index, j] = fillna[j]
    
        scaler = StandardScaler()
        train_df[continuous_features] = scaler.fit_transform(train_df[continuous_features])
        test_df[continuous_features] = scaler.fit_transform(test_df[continuous_features])
        '''
        # Apply preprocessing
        X_train, scaler = preprocess_data(X_train)
        #print('X_train =', X_train[:3, :])
        X_test = scaler.transform(X_test)

        joblib.dump(scaler, f'scaler_run_{i}_fold_{fold}.pkl')  # Save scaler
        '''

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # Create DataLoaders
        train_ds = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        test_ds = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)

        # Model parameters
        input_dim = X_train.shape[1]
        hidden_dim = 64
        num_classes = len(np.unique(y_train))
        num_trees = 5
        tree_depth = 3

        # Model setup
        model = DNDF(input_dim, hidden_dim, num_classes, num_trees, tree_depth).to(device)
        #criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion = FocalLoss(reduction = 'mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train and evaluate
        train_model(model, train_ds, criterion, optimizer, device, epochs=50)
        auprc, auroc, accuracy, recall_value, precision_value, f1 = evaluate_model(model, test_ds, device)

        # Store results
        results_list.append({
            'run': i + 1,
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
