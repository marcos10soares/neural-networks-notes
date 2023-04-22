import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score

class RBF(nn.Module):
    def __init__(self, in_features, out_features, centers=None):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.Parameter(torch.Tensor(out_features, in_features))
        if centers is not None:
            self.centers.data = centers
        else:
            self.centers.data.uniform_(-1, 1)

    def forward(self, x):
        x = x.unsqueeze(1).expand(-1, self.out_features, -1)
        centers = self.centers.unsqueeze(0).expand(x.shape[0], -1, -1)
        diff = x - centers
        dist = torch.norm(diff, dim=2)
        return torch.exp(-1.0 * dist)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

np.random.seed(42)
torch.manual_seed(42)

X, y = make_moons(n_samples=1000, noise=0.1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_data = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
test_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = X_train.shape[1]
rbf_dim = 100
output_dim = 2

model = nn.Sequential(
    RBF(input_dim, rbf_dim),
    nn.Linear(rbf_dim, output_dim)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100

for epoch in range(num_epochs):
    train_model(model, train_loader, criterion, optimizer, device)
    acc = evaluate_model(model, train_loader, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {acc:.2f}")

test_accuracy = evaluate_model(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.2f}")

def plot_decision_boundary(model, X, y):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X_mesh = torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).to(device)

    with torch.no_grad():
        Z = model(X_mesh)
        _, Z = torch.max(Z, 1)

    Z = Z.cpu().numpy()
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50)
    plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    plt.show()

plot_decision_boundary(model, X_train, y_train)
plot_decision_boundary(model, X_test, y_test)

# In this code snippet, we have a plot_decision_boundary function
# that takes the model, input data (X), and labels (y) as input arguments. 
# It creates a meshgrid using the input data and evaluates the model on this grid. 
# The predictions are then reshaped and plotted as a decision boundary 
# using the contourf function from Matplotlib. The training and test data points
#  are also plotted with their true labels.

# When you run this code, you should be able to visualize the decision boundary
#  learned by the RBFN for both the training and test datasets.