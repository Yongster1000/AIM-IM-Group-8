import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# Load and prepare data
train_data = pd.read_csv("Data/train_standard.csv")
test_data = pd.read_csv("Data/test_standard.csv")

# Separate features and labels
X_train = train_data.drop(columns=["target"]) 
y_train = train_data["target"]
X_test = test_data.drop(columns=["target"])
y_test = test_data["target"]

# Normalize feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Create DataLoader
train_ds = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_ds = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)

# Define the neural feature extractor
class NeuralFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(NeuralFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)         #can add more layers if necessary
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# Define a differentiable decision tree with soft routing and trainable leaves
class DNDFTree(nn.Module):
    def __init__(self, depth, input_dim, num_classes):
        super(DNDFTree, self).__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.num_leaf_nodes = 2 ** depth

        # Differentiable decision nodes
        self.decision_layer = nn.Linear(input_dim, self.num_leaf_nodes)   #linear nn layer, to be passed through a activation function (ReLu in this case) to become a probability so as to make soft splits
        
        # Trainable leaf distributions
        self.leaf_distributions = nn.Parameter(torch.rand(self.num_leaf_nodes, num_classes))
        
    def forward(self, x):
        # Soft decision routing at each node
        decision_logits = self.decision_layer(x) #linear NN layer, data type = Tensor, shape: [no. of samples in batch, no. of leaf nodes], obtained by applying linear transformation on embedded values from NN layer, represents likelihood of making a certain decision at decision node
        decision_probs = torch.sigmoid(decision_logits) # logits pass through functions to get a probability, used for soft splits, data type = Tensor, shape: [no. of samples in batch, no. of leaf nodes], values converted to probabilities
        
        # Initialize route probabilities (mu) to start at 1 for each sample
        mu = torch.ones_like(decision_probs[:, :1])  # Start with probability 1

        # Calculate route probabilities layer by layer
        for layer in range(self.depth):
            start_idx = 2 ** layer
            end_idx = 2 ** (layer + 1)
            
            # Concatenate the route probabilities for each split at this layer
            mu = torch.cat([
                mu * decision_probs[:, start_idx:end_idx],              # Path going to "True" side
                mu * (1 - decision_probs[:, start_idx:end_idx])         # Path going to "False" side
            ], dim=1)

        # Combine route probabilities with leaf distributions to get class probabilities
        leaf_distributions = F.softmax(self.leaf_distributions, dim=-1)
        output = torch.matmul(mu, leaf_distributions)
        return output

# DNDF model with a forest of trees
class DNDF(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_trees=5, tree_depth=3):
        super(DNDF, self).__init__()
        self.num_classes = num_classes
        
        # Neural network feature extractor
        self.feature_extractor = NeuralFeatureExtractor(input_dim, hidden_dim)  # N embedding laer
        
        # Forest of trees
        self.trees = nn.ModuleList([DNDFTree(tree_depth, hidden_dim, num_classes) for _ in range(num_trees)])   #Decision Trees (forest) backbone
        
    def forward(self, x):
        # Pass data through the feature extractor
        x = self.feature_extractor(x)
        
        # Aggregate predictions from each tree in the forest
        tree_outputs = [tree(x) for tree in self.trees]
        forest_output = torch.mean(torch.stack(tree_outputs), dim=0)  # Average predictions across trees
        
        return forest_output

# Model parameters
input_dim = X_train.shape[1]  # Number of features in the dataset
hidden_dim = 64
num_classes = len(y_train.unique())  # Number of classes based on unique labels
num_trees = 5
tree_depth = 3

# Initialize the DNDF model
model = DNDF(input_dim, hidden_dim, num_classes, num_trees, tree_depth)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Training the model
train(model, train_ds, criterion, optimizer, epochs=10)

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, accuracy_score, recall_score, f1_score
import numpy as np
import torch

def compute_auprc(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1], pos_label=1)
    auprc = auc(recall, precision)
    return auprc

def compute_auroc(y_true, y_probs):
    return roc_auc_score(y_true, y_probs[:, 1])  # Assuming y_probs[:, 1] for the positive class in binary classification

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average="weighted")  # Use "weighted" for handling class imbalances in multi-class

def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted")  # Use "weighted" to handle class imbalances


def evaluate(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = model(batch_x)
            y_true.extend(batch_y.tolist())
            
            # Get predicted class (highest probability)
            _, predicted = torch.max(output, 1)
            y_pred.extend(predicted.tolist())
            
            # Get predicted probabilities
            y_probs.extend(torch.softmax(output, dim=1).tolist())  # For AUROC and AUPRC

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Calculate metrics
    auprc = compute_auprc(y_true, y_probs)
    auroc = compute_auroc(y_true, y_probs)
    accuracy = compute_accuracy(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)

    # Print or return metrics
    print(f"AUPRC: {auprc:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return auprc, auroc, accuracy, recall, f1

y_pred = evaluate(model, test_ds)