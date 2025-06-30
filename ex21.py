import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image in grayscale
img = cv2.imread("C:\\Users\\user\\OneDrive\\Documents\\Computer vision with openCV\\sample image.jpg", cv2.IMREAD_GRAYSCALE)

# Extended Laplacian kernel including diagonal neighbors
laplacian_ext = np.array([[1, 1, 1],
                          [1, -8, 1],
                          [1, 1, 1]])

# Apply convolution
sharpened = cv2.filter2D(img, -1, laplacian_ext)

# Show results
plt.subplot(1,2,1), plt.title("Original"), plt.imshow(img, cmap='gray'), plt.axis('off')
plt.subplot(1,2,2), plt.title("Sharpened"), plt.imshow(sharpened, cmap='gray'), plt.axis('off')
plt.tight_layout(), plt.show()
