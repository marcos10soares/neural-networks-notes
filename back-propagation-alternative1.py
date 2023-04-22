# In this example, we will use a simple version of the Momentum optimization 
# method to illustrate a different search direction in the error 
# back-propagation process. Momentum is a technique used to accelerate 
# gradient descent in the relevant direction and reduce oscillations.

import torch
import torch.nn as nn
import torch.optim as optim
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
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Train the model
num_epochs = 10000
losses = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the loss curve
plt.plot(range(1, len(losses)+1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss curve using Momentum')
plt.show()


# In this example, we create an MLP model for the XOR problem, just like before. 
# However, this time we are using the SGD optimizer with momentum. 
# The momentum parameter in optim.SGD helps the optimizer to use a different 
# search direction by considering previous gradients. This can help 
# accelerate convergence and reduce oscillations in the loss curve.

# After running this code, you will see the loss curve, which shows the 
# decrease in the loss value over the training iterations. This visualization 
# helps you understand the training progress and convergence of the 
# back-propagation algorithm using a different search direction (Momentum in this case).