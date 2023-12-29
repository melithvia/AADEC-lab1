import numpy as np
import matplotlib.pyplot as plt

# Parameters
f = 400
A = 400.25
B = 399.75
N = 3000
ensemble_size = 6  # Reduced ensemble size for better visualization in a 3x2 grid

# Function to generate random signals
def generate_random_signal():
    k = np.arange(N)
    Wn = np.random.rand(N)
    xn = A * np.cos(2 * np.pi * f * k / N) + B * Wn
    return xn

# Generate ensemble of random signals
ensemble = np.array([generate_random_signal() for _ in range(ensemble_size)])

# 1. Estimate the linear mean as ensemble average
linear_mean = np.mean(ensemble, axis=0)

# 2. Estimate the linear mean and squared linear mean
linear_mean_squared = np.mean(ensemble ** 2, axis=0)

# 3. Estimate the quadratic mean and variance
quadratic_mean = np.sqrt(np.mean(ensemble ** 2, axis=0))
variance = np.var(ensemble, axis=0)

# 4. Plot 1-4 graphically
plt.figure(figsize=(12, 12))

plt.subplot(3, 2, 1)
plt.plot(linear_mean, label='Linear Mean')
plt.title('Linear Mean (Ensemble Average)')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(linear_mean_squared, label='Squared Linear Mean')
plt.title('Squared Linear Mean')
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(quadratic_mean, label='Quadratic Mean')
plt.title('Quadratic Mean')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(variance, label='Variance')
plt.title('Variance')
plt.legend()

plt.subplot(3, 2, 5)
for i in range(ensemble_size):
    plt.plot(np.correlate(ensemble[i], ensemble[i], mode='full'), alpha=0.1, color='blue')

plt.title('Auto-correlation Functions (ACF)')
plt.tight_layout()
plt.show()
