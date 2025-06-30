import cv2
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread("C:\\Users\\user\\OneDrive\\Documents\\Computer vision with openCV\\sample image.jpg", cv2.IMREAD_GRAYSCALE)

# Blur the image
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Subtract blurred from original (unsharp masking)
sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

# Display results
plt.subplot(1,2,1), plt.title("Original"), plt.imshow(img, cmap='gray'), plt.axis('off')
plt.subplot(1,2,2), plt.title("Sharpened"), plt.imshow(sharpened, cmap='gray'), plt.axis('off')
plt.tight_layout(), plt.show()
