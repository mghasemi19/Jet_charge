import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# read the input
final_df = pd.read_csv('uubar.csv')

# Count NaNs in each column
NDIM = len(final_df.keys()) - 1

df_nonan = final_df.copy()
df_nonan = df_nonan.dropna()
#print(df_nonan.isna().sum())
dataset_nonan = df_nonan.values
X = dataset_nonan[:,0:NDIM]
Y = dataset_nonan[:,NDIM]

# Sample data (22498, 68) numpy array
#X = np.random.rand(2498, 68)
#Y = np.random.choice([0, 1], 2498)

# Preprocess the data: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create edge index for a fully connected graph
# For demonstration, a simple fully connected graph is created.
# In practice, you should create an edge_index based on your data's graph structure.
edge_index = torch.tensor(np.array([(i, j) for i in range(len(X)) for j in range(len(X)) if i != j]).T, dtype=torch.long)

# Convert data to torch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float)
Y_tensor = torch.tensor(Y, dtype=torch.long)

# Create a PyTorch Geometric data object
data = Data(x=X_tensor, edge_index=edge_index, y=Y_tensor)

# Split the data into training and testing sets
train_mask, test_mask = train_test_split(np.arange(len(Y)), test_size=0.2, random_state=42)

# Create masks for PyTorch Geometric
data.train_mask = torch.tensor(train_mask, dtype=torch.long)
data.test_mask = torch.tensor(test_mask, dtype=torch.long)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Create the model
model = GCN(in_channels=68, hidden_channels=16, out_channels=2)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing function
def test():
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.size(0)
    return acc

# Training loop
for epoch in range(50):
    loss = train()
    if epoch % 10 == 0:
        acc = test()
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

# Final test accuracy
acc = test()
print(f'Final Test Accuracy: {acc:.4f}')
