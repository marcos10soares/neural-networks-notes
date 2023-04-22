# use PyTorch to create a simple linearly separable dataset, 
# implement a Perceptron model, and visualize the convergence. 
# We'll use matplotlib to plot the data and the decision boundary as the Perceptron converges.

import torch
import matplotlib.pyplot as plt

# Generate linearly separable data
num_samples = 100
data = torch.randn(num_samples, 2)
labels = torch.where(data[:, 1] > data[:, 0], torch.tensor(1.0), torch.tensor(-1.0))

# function to visualize the data and decision boundary
def plot_data_and_boundary(data, labels, weights, bias):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    x = torch.tensor([-2.5, 2.5])
    y = (-bias - weights[0] * x) / weights[1]
    plt.plot(x, y)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Perceptron Decision Boundary")
    plt.show()

# define the Perceptron model and the update rule
class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.weights = torch.zeros(2, requires_grad=False)
        self.bias = torch.zeros(1, requires_grad=False)

    def forward(self, x):
        return torch.sign(torch.matmul(x, self.weights) + self.bias)
    
    def update(self, x, y):
        self.weights += y * x
        self.bias += y

# Instantiate the Perceptron model
model = Perceptron()

# Train the Perceptron and visualize the convergence
num_epochs = 10
for epoch in range(num_epochs):
    num_errors = 0
    for x, y in zip(data, labels):
        prediction = model(x)
        if prediction.item() != y.item():
            model.update(x, y)
            num_errors += 1

    print(f"Epoch {epoch + 1}, Number of errors: {num_errors}")
    plot_data_and_boundary(data, labels, model.weights, model.bias)
    
    # Stop training if there are no errors
    if num_errors == 0:
        break
