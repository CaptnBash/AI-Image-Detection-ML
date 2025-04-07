import os
import settings, model_helper
from settings import *

import cv2
import numpy as np
from progress.bar import Bar
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def compute_histogram(image_path, bins=256):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hist_r = cv2.calcHist([img_rgb], [0], None, [bins], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_rgb], [1], None, [bins], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_rgb], [2], None, [bins], [0, 256]).flatten()
    return np.concatenate([hist_r, hist_g, hist_b])

# get image list
real_images_path_list = [f"{settings.REAL_IMAGES_FOLDER}/{file}" for file in os.listdir(settings.REAL_IMAGES_FOLDER) if file.lower().endswith(('.jpg', '.png'))]
ai_images_path_list = [f"{settings.AI_IMAGES_FOLDER}/{file}" for file in os.listdir(settings.AI_IMAGES_FOLDER) if file.lower().endswith(('.jpg', '.png'))]
images = real_images_path_list + ai_images_path_list

real_images_category = ["REAL"] * len(real_images_path_list)
ai_images_category = ["AI"] * len(ai_images_path_list)
image_categories = real_images_category + ai_images_category

print(f"Given data: {len(images)} images")
print(f"Real images: {len(real_images_path_list)}")
print(f"AI images: {len(ai_images_path_list)}")

if not os.path.exists(HISTOGRAMS_FILE):
    # computing histograms
    histograms = []
    with Bar("Computing histograms...", max=len(images)) as bar:
        for image in images:
            histogram = compute_histogram(image)
            histograms.append(histogram)
            bar.next()
        
    histograms = np.array(histograms)
    np.save(HISTOGRAMS_FILE, histograms)
else:
    histograms = np.load(HISTOGRAMS_FILE)

y = image_categories
X_train, X_test, y_train, y_test = train_test_split(histograms, y, test_size=TEST_SIZE)

clf = model_helper.get_model(X_train, y_train, save=False)

# making predictions
predictions = clf.predict(X_test)

score = accuracy_score(y_test, predictions)
print("Accuracy score: " + str(score))

print(clf.predict([compute_histogram("image.jpg")]))