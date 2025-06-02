import numpy as np
import cv2
import matplotlib.pyplot as plt

# load the image
img = cv2.imread('dog.jpg', 0)

# Compute FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Create a mask with a low-pass filter
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.uint8)

mask[crow-30:crow+30, ccol-30:ccol+30] = 1  # Creating a simple low-pass filter mask

# uncomment to test with other numbers
# mask[crow-50:crow+50, ccol-50:ccol+50] = 1  
# mask[crow-100:crow+100, ccol-100:ccol+100] = 1

# Apply mask and inverse FFT
fshift_masked = fshift * mask
f_ishift = np.fft.ifftshift(fshift_masked)
# The result is a complex number array
image_filtered = np.fft.ifft2(f_ishift)

# we take the magnitude (absolute value) to get a real image we can display.
image_filtered = np.abs(image_filtered)

# Display the filtered image
plt.imshow(image_filtered, cmap='gray')
plt.title('Low-Pass Filtered Image')
plt.show()

# This method is used to remove high-frequency components, leading to image blurring.
