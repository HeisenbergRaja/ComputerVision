import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image in grayscale
img = cv2.imread("C:\\Users\\user\\OneDrive\\Documents\\Computer vision with openCV\\sample image.jpg", cv2.IMREAD_GRAYSCALE)

# Laplacian mask with positive center (5) and -1 at four neighbors
laplacian_positive = np.array([[ 0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]])

# Apply the filter
sharpened = cv2.filter2D(img, -1, laplacian_positive)

# Show results
plt.subplot(1,2,1), plt.title("Original"), plt.imshow(img, cmap='gray'), plt.axis('off')
plt.subplot(1,2,2), plt.title("Sharpened"), plt.imshow(sharpened, cmap='gray'), plt.axis('off')
plt.tight_layout(), plt.show()
