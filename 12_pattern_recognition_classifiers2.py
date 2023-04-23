import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate sample data for healthy and sick patients
np.random.seed(42)
healthy_temp = np.random.normal(loc=36.5, scale=0.5, size=1000)
sick_temp = np.random.normal(loc=38, scale=0.5, size=1000)

# Estimate the parameters for the Gaussian distributions
healthy_mean, healthy_std = np.mean(healthy_temp), np.std(healthy_temp)
sick_mean, sick_std = np.mean(sick_temp), np.std(sick_temp)

# Calculate the Gaussian probability density functions
x = np.linspace(35, 40, 1000)
healthy_pdf = norm.pdf(x, healthy_mean, healthy_std)
sick_pdf = norm.pdf(x, sick_mean, sick_std)

# Calculate the Bayesian threshold
threshold = (healthy_std**2 * np.log(sick_std / healthy_std) -
             sick_std**2 * np.log(healthy_std / sick_std) +
             2 * (sick_mean * healthy_std**2 - healthy_mean * sick_std**2)) / (healthy_std**2 - sick_std**2)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.hist(healthy_temp, bins=30, alpha=0.5, label='Healthy', density=True)
plt.hist(sick_temp, bins=30, alpha=0.5, label='Sick', density=True)
plt.plot(x, healthy_pdf, label='Healthy PDF', color='blue')
plt.plot(x, sick_pdf, label='Sick PDF', color='red')
plt.axvline(x=threshold, color='black', linestyle='--', label=f'Bayesian Threshold: {threshold:.2f}')

# Set labels
plt.xlabel('Temperature')
plt.ylabel('Density')
plt.legend()
plt.title("Baysean Threshold")

# Set x-axis limits
plt.xlim(34.5,40)
plt.show()