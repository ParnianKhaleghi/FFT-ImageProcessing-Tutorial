import numpy as np
import cv2
import matplotlib.pyplot as plt

# load the image
#  actually read the image in grayscale (single channel).
img = cv2.imread('dog.jpg', 0)

# Compute FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Compute magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Display the original image and its magnitude spectrum
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Magnitude Spectrum')
plt.show()

# This technique is used to convert an image from the spatial domain to the frequency domain.