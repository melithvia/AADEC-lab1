import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal.windows import hann, flattop

# Parameters
f1 = 400 # Hz  
f2 = 400.25 # Hz  
f3 = 399.75 # Hz  
fs = 600 # Hz 
amp_max = 3
N = 3000 
k = np.arange(N)

# Generate three sine signals
x1 = amp_max * np.sin(2 * np.pi * f1 / fs * k)
x2 = amp_max * np.sin(2 * np.pi * f2 / fs * k)
x3 = amp_max * np.sin(2 * np.pi * f3 / fs * k)

# Compute DFT spectra
X1 = fft(x1)
X2 = fft(x2)
X3 = fft(x3)

# Frequency axis
frequencies = np.fft.fftfreq(N, 1/fs)
omega = 2 * np.pi * frequencies

# Compute windowed DFT spectra
wrect = np.ones(N)
whann = hann(N, sym=False)
wflattop = flattop(N, sym=False)

X1_rect = fft(x1 * wrect)
X2_rect = fft(x2 * wrect)
X3_rect = fft(x3 * wrect)

X1_hann = fft(x1 * whann)
X2_hann = fft(x2 * whann)
X3_hann = fft(x3 * whann)

X1_flattop = fft(x1 * wflattop)
X2_flattop = fft(x2 * wflattop)
X3_flattop = fft(x3 * wflattop)

# Normalize the spectra
X1_norm = np.abs(X1) / N
X2_norm = np.abs(X2) / N
X3_norm = np.abs(X3) / N

X1_rect_norm = np.abs(X1_rect) / N
X2_rect_norm = np.abs(X2_rect) / N
X3_rect_norm = np.abs(X3_rect) / N

X1_hann_norm = np.abs(X1_hann) / N
X2_hann_norm = np.abs(X2_hann) / N
X3_hann_norm = np.abs(X3_hann) / N

X1_flattop_norm = np.abs(X1_flattop) / N
X2_flattop_norm = np.abs(X2_flattop) / N
X3_flattop_norm = np.abs(X3_flattop) / N

# Plotting
plt.figure(figsize=(12, 8))

# Plotting DFT Spectra
plt.subplot(2, 1, 1)
plt.plot(omega, 20 * np.log10(X1_norm), label='F1')
plt.plot(omega, 20 * np.log10(X2_norm), label='F2')
plt.plot(omega, 20 * np.log10(X3_norm), label='F3')
plt.title('Normalized DFT Spectra')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid()

# Plotting Windowed DFT Spectra
plt.subplot(2, 1, 2)
plt.plot(omega, 20 * np.log10(X1_rect_norm), label='F1 (Rectangular)')
plt.plot(omega, 20 * np.log10(X2_rect_norm), label='F2 (Rectangular)')
plt.plot(omega, 20 * np.log10(X3_rect_norm), label='F3 (Rectangular)')
plt.plot(omega, 20 * np.log10(X1_hann_norm), label='F1 (Hann)')
plt.plot(omega, 20 * np.log10(X2_hann_norm), label='F2 (Hann)')
plt.plot(omega, 20 * np.log10(X3_hann_norm), label='F3 (Hann)')
plt.plot(omega, 20 * np.log10(X1_flattop_norm), label='F1 (Flat Top)')
plt.plot(omega, 20 * np.log10(X2_flattop_norm), label='F2 (Flat Top)')
plt.plot(omega, 20 * np.log10(X3_flattop_norm), label='F3 (Flat Top)')
plt.title('Normalized Windowed DFT Spectra')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()