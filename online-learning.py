# demonstrate online learning with adapting only linear weights 
# for a Multilayer Perceptron (MLP) model. In this example, we'll 
# pre-train the hidden layers using the entire dataset, and then 
# update only the output layer's weights with online learning.

import torch
import torch.nn as nn
import torch.optim as optim
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

# Pre-train the MLP with the entire dataset
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Freeze all layers except the output layer
for param in model.layer1.parameters():
    param.requires_grad = False

# Online learning: Adapting only linear weights (output layer)
online_optimizer = optim.Adam(model.layer2.parameters(), lr=0.01)

for i in range(len(X)):
    online_optimizer.zero_grad()
    output = model(X[i])
    loss = criterion(output, y[i])
    loss.backward()
    online_optimizer.step()

# Visualization
with torch.no_grad():
    y_pred = model(X)

plt.scatter(X, y, label='True data')
plt.scatter(X, y_pred, label='Predictions', color='red', marker='.')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


# In this example, we first create a simple dataset and define 
# the Multilayer Perceptron (MLP) architecture. We pre-train the 
# entire network using the whole dataset for 1000 epochs. Then, 
# we freeze the hidden layer's weights (non-linear weights) and 
# update only the output layer's weights (linear weights) using 
# online learning, processing one data point at a time. Finally, 
# we visualize the true data and the model's predictions. Note 
# that the results may vary due to the randomness involved in the
# dataset and model initialization.