# quasi-Newton method using the L-BFGS-B optimizer
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

# Flatten the initial weights of the model
initial_weights = np.concatenate([p.detach().numpy().ravel() for p in model.parameters()])

# Define the objective function
losses = []

def objective_function(w, model, X, y):
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            p.copy_(torch.tensor(w[i]))

    y_pred = model(X).squeeze()
    residuals = (y.squeeze() - y_pred).detach().numpy()
    loss = 0.5 * np.sum(residuals**2)
    losses.append(loss)
    return loss

# Perform BFGS optimization
result = minimize(fun=objective_function, x0=initial_weights, args=(model, X, y), method='L-BFGS-B')

# Assign the optimized weights to the model
for i, p in enumerate(model.parameters()):
    p.data = torch.tensor(result.x[i])

# Plot the loss curve
plt.plot(range(1, len(losses)+1), losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss curve using L-BFGS-B')
plt.show()

# In this example, we use the L-BFGS-B optimizer from the Scipy library 
# to optimize the model's parameters by minimizing the objective function. 
# The L-BFGS-B algorithm is a quasi-Newton method that is suitable 
# for a broader range of problems than the Gauss-Newton method.

# You can visualize the loss curve by plotting the losses list, 
# which shows the decrease in the loss value over the optimization iterations.