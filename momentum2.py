import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# learning rate helps increase/reduce oscilations in MSE
epochs = 1000
learning_rate = 0.05

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

# Train the MLP using SGD without momentum
model_no_momentum = MLP()
criterion = nn.MSELoss()
optimizer_no_momentum = optim.SGD(model_no_momentum.parameters(), lr=learning_rate)

mse_no_momentum = []
for epoch in range(epochs):
    optimizer_no_momentum.zero_grad()
    outputs = model_no_momentum(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer_no_momentum.step()
    mse_no_momentum.append(loss.item())

# Train the MLP using SGD with momentum
model_momentum = MLP()
optimizer_momentum = optim.SGD(model_momentum.parameters(), lr=learning_rate, momentum=0.9)

mse_momentum = []
for epoch in range(epochs):
    optimizer_momentum.zero_grad()
    outputs = model_momentum(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer_momentum.step()
    mse_momentum.append(loss.item())

# Visualization
plt.plot(mse_no_momentum, label='Without momentum')
plt.plot(mse_momentum, label='With momentum', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()
