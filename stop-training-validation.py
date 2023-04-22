import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Create a simple dataset
X = torch.linspace(-1, 1, 100).reshape(-1, 1)
y = X.pow(2) + 0.2 * torch.rand(X.size())

# Define the Multilayer Perceptron (MLP) architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = self.layer2(x)
        return x

# Training function
def train(model, optimizer, criterion, X_train, y_train, epochs=1000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

# Evaluation function
def evaluate(model, X_test, y_test):
    with torch.no_grad():
        y_pred = model(X_test)
        mse = mean_squared_error(y_test, y_pred)
    return mse

# Normal (Holdout) Validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

train(model, optimizer, criterion, X_train, y_train)
normal_mse = evaluate(model, X_test, y_test)
print(f"Normal (Holdout) Validation MSE: {normal_mse}")

# # Visualization
with torch.no_grad():
    y_pred_normal = model(X)

# plt.title('MSE - K-Fold vs Normal Validation')
# plt.scatter(X, y, label='True data')
# plt.scatter(X, y_pred, label='Predictions', color='red', marker='.')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.legend()
# plt.show()

# K-Fold Cross-Validation
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

mse_list = []

for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    train(model, optimizer, criterion, X_train, y_train)
    mse = evaluate(model, X_test, y_test)
    mse_list.append(mse)

k_fold_mse = np.mean(mse_list)
print(f"K-Fold Cross-Validation MSE: {k_fold_mse}")

# Visualization
with torch.no_grad():
    y_pred = model(X)

plt.title('MSE - K-Fold vs Normal Validation')
plt.scatter(X, y, label='True data')
plt.scatter(X, y_pred_normal, label='Normal Predictions', color='#4421af', marker='.')
plt.scatter(X, y_pred, label='K-Fold Predictions', color='#b30000', marker='.')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
