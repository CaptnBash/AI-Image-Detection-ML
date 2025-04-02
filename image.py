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

# pick a random images
indices = random.sample(range(len(images)), 10000)
rand_images = [images[i] for i in indices]
rand_categories = [image_categories[i] for i in indices]

# computing histograms
histograms = []
with Bar("Computing histograms...", max=len(indices)) as bar:
    for image in rand_images:
        histogram = compute_histogram(image)
        histograms.append(histogram)
        bar.next()
        
y = rand_categories
# np.save("histograms.npy", histograms)

X_train, X_test, y_train, y_test = train_test_split(histograms, y, test_size=0.2)
clf = RandomForestClassifier().fit(X_train, y_train)

predictions = clf.predict(X_test)

score = accuracy_score(y_test, predictions)
print(score)