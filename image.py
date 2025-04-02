import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from progress.bar import Bar
import os, random

def compute_histogram(image_path, bins=256):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hist_r = cv2.calcHist([img_rgb], [0], None, [bins], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_rgb], [1], None, [bins], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_rgb], [2], None, [bins], [0, 256]).flatten()
    return np.concatenate([hist_r, hist_g, hist_b])

# get image list
real_images = [f"REAL/{file}" for file in os.listdir("REAL") if file.lower().endswith(('.jpg', '.png'))]
ai_images = [f"AI/{file}" for file in os.listdir("AI") if file.lower().endswith(('.jpg', '.png'))]
images = real_images + ai_images

real_images_category = ["REAL"] * len(real_images)
ai_images_category = ["AI"] * len(ai_images)
image_categories = real_images_category + ai_images_category

print(f"Given data: {len(images)} images")
print(f"Real images: {len(real_images)}")
print(f"AI images: {len(ai_images)}")

histograms_file = "histograms.npy"
if not os.path.exists(histograms_file):
    # computing histograms
    histograms = []
    with Bar("Computing histograms...", max=len(images)) as bar:
        for image in images:
            histogram = compute_histogram(image)
            histograms.append(histogram)
            bar.next()
        
    histograms = np.array(histograms)
    np.save(histograms_file, histograms)

# training model
print("\ntraining model...")
histograms = np.load(histograms_file)
y = image_categories
X_train, X_test, y_train, y_test = train_test_split(histograms, y, test_size=0.2)
clf = RandomForestClassifier().fit(X_train, y_train)

predictions = clf.predict(X_test)

score = accuracy_score(y_test, predictions)
print("Accuracy score: " + str(score))