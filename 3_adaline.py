# Adaline model and visualize its decision boundary. 
# Use a linearly separable dataset 
# Train the Adaline model and visualize the convergence

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
    plt.title("Adaline Decision Boundary")
    plt.show()


class Adaline(torch.nn.Module):
    def __init__(self, learning_rate):
        super(Adaline, self).__init__()
        self.weights = torch.zeros(2, requires_grad=False)
        self.bias = torch.zeros(1, requires_grad=False)
        self.learning_rate = learning_rate

    def forward(self, x):
        return torch.matmul(x, self.weights) + self.bias
    
    def predict(self, x):
        return torch.sign(self.forward(x))

    def update(self, x, y):
        output = self.forward(x)
        error = y - output
        self.weights += self.learning_rate * error * x
        self.bias += self.learning_rate * error

# Instantiate the Adaline model
learning_rate = 0.01
model = Adaline(learning_rate)

# Train the Adaline model and visualize the convergence
num_epochs = 10
for epoch in range(num_epochs):
    num_errors = 0
    for x, y in zip(data, labels):
        prediction = model.predict(x)
        if prediction.item() != y.item():
            model.update(x, y)
            num_errors += 1

    print(f"Epoch {epoch + 1}, Number of errors: {num_errors}")
    plot_data_and_boundary(data, labels, model.weights, model.bias)
    
    # Stop training if there are no errors
    if num_errors == 0:
        break
