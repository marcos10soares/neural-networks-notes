# quasi-Newton method
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Create the dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

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

input_size = 2
hidden_size = 4
output_size = 1
model = MLP(input_size, hidden_size, output_size)
criterion = nn.MSELoss()

def closure():
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    return loss


# train the model using the L-BFGS optimizer from PyTorch
optimizer = torch.optim.LBFGS(model.parameters(), lr=1)

losses = []
num_iterations = 20
for i in range(num_iterations):
    optimizer.step(closure)
    with torch.no_grad():
        y_pred = model(X)
        loss = criterion(y_pred, y)
        losses.append(loss.item())
        print(f"Iteration {i+1}, Loss: {loss.item()}")

# Plot the loss curve
plt.plot(range(1, num_iterations+1), losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss curve using L-BFGS')
plt.show()

# In this example, we use the L-BFGS optimizer from PyTorch's optimization module. 
# The closure function computes the loss and its gradients for the optimizer. 
# We run the training loop for a fixed number of iterations, updating the 
# model parameters using the L-BFGS optimizer. The loss values are recorded in 
# the losses list and plotted at the end to visualize the training progress.

# After running this code, you will see the loss curve, which shows the decrease 
# in the loss value over the training iterations. This visualization helps you 
# understand the training progress and convergence of the L-BFGS optimizer.