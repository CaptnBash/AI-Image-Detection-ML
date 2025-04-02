import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def compute_histogram(image_path, bins=256):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hist_r = cv2.calcHist([img_rgb], [0], None, [bins], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_rgb], [1], None, [bins], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_rgb], [2], None, [bins], [0, 256]).flatten()
    return np.concatenate([hist_r, hist_g, hist_b])

image_paths = os.listdir("images")

histograms = np.array([compute_histogram(f"images/{path}") for path in image_paths])
y = ["real"] * len(histograms)
# np.save("histograms.npy", histograms)

X_train, X_test, y_train, y_test = train_test_split(histograms, y, test_size=0.2)
clf = RandomForestClassifier().fit(X_train, y_train)

predictions = clf.predict(X_test)

score = accuracy_score(y_test, predictions)
print(score)