import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate example data
np.random.seed(0)
n_samples = 1000

healthy_temp = np.random.normal(36.5, 0.5, n_samples // 2)
sick_temp = np.random.normal(38.0, 0.5, n_samples // 2)
temperatures = np.concatenate([healthy_temp, sick_temp])

# Create target labels
labels = np.zeros(n_samples)
labels[n_samples // 2:] = 1

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(temperatures.reshape(-1, 1), labels, test_size=0.3, random_state=0)

# Train the SVM with a radial basis function kernel
clf = SVC(kernel='rbf', C=1e5)
clf.fit(X_train, y_train)

# Make predictions and calculate the mean squared error
y_pred = clf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)

# Plot the results
xx = np.linspace(34, 40, 1000).reshape(-1, 1)
yy = clf.predict(xx)

healthy = plt.scatter(X_train[:, 0], y_train, c=y_train, edgecolors='k', marker='o', s=100, alpha=0.8, cmap=plt.cm.coolwarm)
sick = plt.scatter(X_test[:, 0], y_test, c=y_test, edgecolors='k', marker='^', s=100, alpha=0.8, cmap=plt.cm.coolwarm)

plt.plot(xx, yy, '-k', lw=2)
plt.fill_between(xx.ravel(), -0.2, yy, color='red', alpha=0.1)
plt.fill_between(xx.ravel(), yy, 1.2, color='blue', alpha=0.1)

# Find the decision boundary value
boundary_indices = np.where(np.abs(yy[:-1] - yy[1:]) > 1e-5)[0]
if len(boundary_indices) > 1:
    boundary_idx = boundary_indices[1]
else:
    boundary_idx = boundary_indices[0]
boundary_value = xx[boundary_idx][0]
print('decision boundary:', boundary_value)


plt.annotate(f'Decision boundary: {boundary_value:.2f}',
                xy=(boundary_value, 0.5),
                xytext=(boundary_value - 1.5, 0.7),
                fontsize=12,
                arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.xlabel('Temperature')
plt.ylabel('Label (0: Healthy, 1: Sick)')
plt.title('Kernel SVM for Patient Temperature Problem')

plt.ylim([-0.2, 1.2])
plt.xlim(34.5, 40)
plt.legend((healthy, sick), ('Healthy', 'Sick'))
plt.show()
