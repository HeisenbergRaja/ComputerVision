import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread("C:\\Users\\user\\OneDrive\\Documents\\Computer vision with openCV\\image1.png", cv2.IMREAD_GRAYSCALE)

# Choose A ≥ 1 (e.g., A = 1.5 for high-boost)
A = 1.5

# High-boost kernel (center = A + 8, others = -1)
high_boost_kernel = np.array([[-1, -1, -1],
                              [-1, A + 8, -1],
                              [-1, -1, -1]])

# Apply filter
sharpened = cv2.filter2D(img, -1, high_boost_kernel)

# Display result
plt.subplot(1,2,1), plt.title("Original"), plt.imshow(img, cmap='gray'), plt.axis('off')
plt.subplot(1,2,2), plt.title("High-Boost Sharpened"), plt.imshow(sharpened, cmap='gray'), plt.axis('off')
plt.tight_layout(), plt.show()
