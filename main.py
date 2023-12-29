import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def idft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)

    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(1j * 2 * np.pi * k * n / N)

    return x / N

X = [6, 8, 2, 4, 3, 4, 5, 0, 0, 0]
x = idft(X)
print("Time domain signal:", x)

# Plotting
plt.figure(figsize=(12, 6))

# Plot real part
plt.subplot(1, 2, 1)
plt.stem(np.real(x))
plt.title("Real Part of IDFT")
plt.xlabel("Sample index")
plt.ylabel("Amplitude")

# Plot imaginary part
plt.subplot(1, 2, 2)
plt.stem(np.imag(x))
plt.title("Imaginary Part of IDFT")
plt.xlabel("Sample index")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

