import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Generate sample data for healthy and sick patients
np.random.seed(42)
num_samples = 1000
healthy_temp = np.random.normal(loc=36.5, scale=0.5, size=num_samples)
sick_temp = np.random.normal(loc=38, scale=0.5, size=num_samples)

# Estimate the parameters for the Gaussian distributions
healthy_mean, healthy_std = np.mean(healthy_temp), np.std(healthy_temp)
sick_mean, sick_std = np.mean(sick_temp), np.std(sick_temp)

def mahalanobis_distance(x, mean, variance):
    diff = x - mean
    inv_variance = 1 / variance
    return np.sqrt(diff * inv_variance * diff)

# Calculate variances
healthy_variance = np.var(healthy_temp)
sick_variance = np.var(sick_temp)

# Combine data
all_data = np.concatenate([healthy_temp, sick_temp])
true_labels = np.concatenate([np.zeros(len(healthy_temp)), np.ones(len(sick_temp))])

predicted_labels = []
error_count = 0

for data_point, true_label in zip(all_data, true_labels):
    healthy_distance = mahalanobis_distance(data_point, healthy_mean, healthy_variance)
    sick_distance = mahalanobis_distance(data_point, sick_mean, sick_variance)
    
    if healthy_distance < sick_distance:
        predicted_label = 0
    else:
        predicted_label = 1
    
    predicted_labels.append(predicted_label)
    
    if predicted_label != true_label:
        error_count += 1

error_rate = error_count / len(all_data)
minimum_error_rate = min(error_rate, 1 - error_rate)

bayesian_threshold = (healthy_mean + sick_mean) / 2

# Bayesian threshold classifier
predicted_labels_bayesian = []
error_count_bayesian = 0

for data_point, true_label in zip(all_data, true_labels):
    if data_point < bayesian_threshold:
        predicted_label_bayesian = 0
    else:
        predicted_label_bayesian = 1
    
    predicted_labels_bayesian.append(predicted_label_bayesian)
    
    if predicted_label_bayesian != true_label:
        error_count_bayesian += 1

error_rate_bayesian = error_count_bayesian / len(all_data)

print(f"Error rate (Mahalanobis): {error_rate:.2f}")
print(f"Error rate (Bayesian): {error_rate_bayesian:.2f}")
print(f"Minimum error rate: {minimum_error_rate:.2f}")

# The error rates for Mahalanobis and Bayesian classifiers 
# appear to be the same in this particular example because 
# the class distributions are assumed to be Gaussian with 
# equal variances. Under these conditions, both classifiers 
# would produce similar decisions.

# However, the minimum error rate represents the lowest possible
# error rate achievable under the given assumptions about the
# class distributions. In practice, the minimum error rate might
# not be achieved by any specific classifier, and it's possible 
# that the error rates of different classifiers may vary.

# In this specific example, it seems that both the Mahalanobis 
# distance classifier and the Bayesian threshold classifier 
# perform well, achieving the minimum error rate. It's essential
# to understand that this might not always be the case, and the
# performance of different classifiers may vary depending on the 
# problem and the assumptions made about the data.



## Decision Surface

# Combine the data and labels
X = np.hstack((healthy_temp, sick_temp)).reshape(-1, 1)
y = np.hstack((np.zeros(num_samples), np.ones(num_samples)))

# Train the logistic regression model
clf = LogisticRegression()
clf.fit(X, y)

# Define a function to calculate the decision boundary
def decision_boundary(x, clf):
    return 0.5 - clf.predict_proba(x.reshape(-1, 1))[:, 1]

# Find the decision boundary temperature
boundary_temp = np.linspace(30, 40, 1000)
boundary_index = np.argmin(np.abs(decision_boundary(boundary_temp, clf)))

# Visualize the decision surface
plt.hist(healthy_temp, bins=30, alpha=0.5, label='Healthy')
plt.hist(sick_temp, bins=30, alpha=0.5, label='Sick')
plt.axvline(x=boundary_temp[boundary_index], color='r', linestyle='--', label=f'Decision boundary (T): {boundary_temp[boundary_index]:.2f}')
plt.legend()
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.title('Decision Surface')
plt.xlim(34.5, 40)
plt.show()