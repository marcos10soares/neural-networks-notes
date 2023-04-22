# Levenberg-Marquardt
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# use the Levenberg-Marquardt algorithm from the SciPy library to train the model
from scipy.optimize import least_squares

# Create a synthetic dataset
num_samples = 100
X = torch.tensor(np.random.randint(0, 2, size=(num_samples, 2)), dtype=torch.float32)
y = (X[:, 0] != X[:, 1]).float().unsqueeze(-1)

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

# Flatten model parameters
initial_weights = []
for p in model.parameters():
    initial_weights.extend(p.detach().numpy().flatten())
initial_weights = np.array(initial_weights)


# In this example, we define the objective_function that computes the residuals 
# (the difference between the target and predicted values). 
# We then use the least_squares function from SciPy's optimize module, 
# passing the Levenberg-Marquardt method ('lm') as the optimization method.

# To visualize the training progress, we can modify the objective_function to 
# record the loss values during the optimization process
losses = []

def objective_function(w, model, X, y):
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            p.copy_(torch.tensor(w[i]))

    y_pred = model(X).squeeze()
    residuals = (y.squeeze() - y_pred).detach().numpy()
    loss = 0.5 * np.sum(residuals**2)
    losses.append(loss)
    return residuals


# Train the model using the Levenberg-Marquardt algorithm
result = least_squares(fun=objective_function, x0=initial_weights, args=(model, X, y), method='lm')

# Plot the loss curve
plt.plot(range(1, len(losses)+1), losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss curve using Levenberg-Marquardt')
plt.show()

# After running this code, you will see the loss curve, which shows the decrease 
# in the loss value over the training iterations. This visualization helps you 
# understand the training progress and convergence of the Levenberg-Marquardt algorithm


# The spikes in the loss curve can be attributed to several factors, 
# including the optimization process, the synthetic dataset used, 
# and the model's initialization. Here are some explanations for the spikes:

#     - Optimization process: The Levenberg-Marquardt algorithm is an 
#       iterative optimization method that updates the model parameters 
#       to minimize the sum of squared residuals. During the optimization process, 
#       the algorithm may take some steps that temporarily increase the loss 
#       before finding a better direction that reduces the loss. This can result 
#       in spikes in the loss curve.

#     - Synthetic dataset: The dataset we generated for this example is not a 
#       perfect representation of the XOR problem. With random samples, 
#       the dataset may contain some noise, making the optimization more challenging 
#       and leading to spikes in the loss curve.

#     - Model initialization: The initial values of the model parameters can have 
#       a significant impact on the optimization process. If the model starts 
#       with a poor initialization, it may take some time for the optimization algorithm 
#       to find a suitable direction to minimize the loss, resulting in spikes in the loss curve.

# To potentially reduce the spikes in the loss curve, you could try the following:

#     - Modify the synthetic dataset: Generate a dataset that better represents
#       the XOR problem or use a real-world dataset that has a larger number of 
#       samples and potentially smoother relationships between input and output.

#     - Initialize the model with better values: Experiment with different 
#       initialization techniques for the model parameters, such as Xavier/Glorot 
#       or He initialization, to potentially improve convergence.

#     - Tune the optimization algorithm: Adjust the hyperparameters of the 
#       Levenberg-Marquardt algorithm, such as the damping factor, to potentially 
#       improve convergence and reduce the spikes in the loss curve.

# Keep in mind that the presence of spikes in the loss curve does not 
# necessarily mean the optimization process has failed. It's essential 
# to look at the overall trend of the loss curve and the final performance 
# of the model on the validation or test set to evaluate the optimization's success.