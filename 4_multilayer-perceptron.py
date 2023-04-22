# Multilayer Perceptron (MLP) with PyTorch to classify a nonlinear dataset. 
# We'll use a single hidden layer and a sigmoid activation function. 
# After training the MLP using backpropagation, we'll visualize the decision boundary.

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

num_samples = 100

# Generate XOR dataset
data = torch.randn(num_samples, 2)
labels = torch.where((data[:, 0] * data[:, 1]) > 0, torch.tensor(1.0), torch.tensor(0.0))


def plot_data_and_boundary(data, labels, model):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    
    x1 = np.linspace(data[:, 0].min() - 0.5, data[:, 0].max() + 0.5, 100)
    x2 = np.linspace(data[:, 1].min() - 0.5, data[:, 1].max() + 0.5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    X = torch.tensor(np.column_stack((X1.ravel(), X2.ravel())), dtype=torch.float32)
    Y = model(X).detach().numpy()
    Y = np.argmax(Y, axis=1)
    Y = Y.reshape(X1.shape)
    
    plt.contourf(X1, X2, Y, alpha=0.3, cmap='viridis')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("MLP Decision Boundary")
    plt.show()



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        return x


# Instantiate the MLP model
input_size = 2
hidden_size = 4
output_size = 2
model = MLP(input_size, hidden_size, output_size)

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the MLP model using backpropagation
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels.long())
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

plot_data_and_boundary(data, labels, model)
