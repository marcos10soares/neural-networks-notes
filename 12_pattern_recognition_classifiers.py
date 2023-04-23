import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate synthetic data (temperatures)
np.random.seed(42)
healthy_temps = np.random.normal(loc=36.5, scale=0.5, size=200)
sick_temps = np.random.normal(loc=38.5, scale=0.5, size=200)

# Combine healthy and sick temperatures and create labels
temperatures = np.concatenate([healthy_temps, sick_temps])
labels = np.array([0] * 200 + [1] * 200)

# Reshape data
temperatures = temperatures.reshape(-1, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(temperatures, labels, test_size=0.2, random_state=42)

# Train the Gaussian Naive Bayes classifier
clf = GaussianNB()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.matshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.show()