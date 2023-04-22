# example of how to compute the Jacobian matrix of a neural network 
# with respect to its inputs using PyTorch. 
# We will use the Multilayer Perceptron (MLP) model from the previous example
import torch
import torch.nn as nn
import torch.autograd as autograd

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
output_size = 2
model = MLP(input_size, hidden_size, output_size)

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = model(x)

# Compute the Jacobian matrix of y with respect to x
jacobian = torch.zeros(output_size, input_size)
for i in range(output_size):
    gradients = autograd.grad(outputs=y[i], inputs=x, retain_graph=True, create_graph=True)[0]
    jacobian[i, :] = gradients

print("Input:", x)
print("Output:", y)
print("Jacobian matrix:")
print(jacobian)

# In this example, we first create a tensor x with the input values 
# and set requires_grad=True to enable gradient computation. 
# We then compute the output y of the model for the given input. 
# To compute the Jacobian matrix, we loop over each output element 
# and use the autograd.grad() function to compute the gradients of 
# the output with respect to the input variables. 
# The gradients are then stored in the jacobian matrix.

# This code will output the input, output, and the c
# orresponding Jacobian matrix for a single input to the MLP model.