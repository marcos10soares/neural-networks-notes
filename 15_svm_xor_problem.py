# considering the following XOR Problem:
# - assume that a polynomial inner-product kernel is used K(x, x_i) = (1 + (x^T) * x)^2
# - The input data and the target vector are as per the following table:
#     |     x    |  t  |
#     | (-1, -1) | -1  |
#     | (-1, +1) | +1  |
#     | (+1,- 1) | +1  |
#     | (+1, +1) | -1  |

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Custom kernel function
def polynomial_kernel(X, Y):
    return (1 + np.dot(X, Y.T)) ** 2

# Input data
X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y = np.array([-1, 1, 1, -1])

# Calculate kernel matrix
kernel_matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        kernel_matrix[i, j] = polynomial_kernel(X[i], X[j])

print(kernel_matrix)

# Train an SVM with the custom polynomial kernel
clf = SVC(kernel=polynomial_kernel, C=1e5)
clf.fit(X, y)

# Plot the decision boundary
xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100, cmap=plt.cm.coolwarm)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

# Add labels and title
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM with Polynomial Kernel for XOR Problem')

# Add a legend
plt.legend(*scatter.legend_elements(), title="Classes")

plt.show()
