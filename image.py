import cv2
import matplotlib.pyplot as plt

img = cv2.imread('img.JPG')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])

# Plot histograms
plt.plot(hist_r, color='red', label='Red')
plt.plot(hist_g, color='green', label='Green')
plt.plot(hist_b, color='blue', label='Blue')
plt.legend()
plt.show(block=True)