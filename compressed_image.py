import numpy as np
import cv2
import matplotlib.pyplot as plt

# load the image
img = cv2.imread('dog.jpg', 0)
# Compute FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# finds the maximum magnitude in the frequency domain
# We keep only components greater than 10% of this maximum
# If a frequency component is smaller than this, we’ll remove it (set it to 0).
threshold = 0.1 * np.max(np.abs(fshift))

# Frequencies larger than the threshold are kept.
# Others are set to 0 (removed/compressed).
fshift_compressed = np.where(np.abs(fshift) > threshold, fshift, 0)

# Inverse FFT to reconstruct the image
f_ishift = np.fft.ifftshift(fshift_compressed)
compressed_image = np.fft.ifft2(f_ishift)
compressed_image = np.abs(compressed_image)

# Display the compressed image
plt.imshow(compressed_image, cmap='gray')
plt.title('Compressed Image')
plt.show()

# When applying FFT compression in a image we may see a strange blurring effect this is 
# because we manipulate the frequency components of the image during the compression process.
# By transforming an image into its frequency components less significant frequencies 
# are discarded leading compression.