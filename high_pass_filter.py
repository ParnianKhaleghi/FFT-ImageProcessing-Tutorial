import numpy as np
import cv2
import matplotlib.pyplot as plt

# load the image
img = cv2.imread('dog.jpg', 0)

# Compute FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Create a high-pass filter mask
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0  # Creating a simple high-pass filter mask

# Apply mask and inverse FFT
fshift_masked = fshift * mask
f_ishift = np.fft.ifftshift(fshift_masked)
image_filtered = np.fft.ifft2(f_ishift)
image_filtered = np.abs(image_filtered)

# Display the filtered image
plt.imshow(image_filtered, cmap='gray')
plt.title('High-Pass Filtered Image')
plt.show()