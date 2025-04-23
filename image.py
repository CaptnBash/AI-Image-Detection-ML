import os, time
import model_helper
from settings import *

import cv2
import numpy as np
from progress.bar import Bar
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def compute_histogram(image_path, bins=256):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist_r = cv2.calcHist([img_rgb], [0], None, [bins], [0, 256]).flatten()
        hist_g = cv2.calcHist([img_rgb], [1], None, [bins], [0, 256]).flatten()
        hist_b = cv2.calcHist([img_rgb], [2], None, [bins], [0, 256]).flatten()
        return np.concatenate([hist_r, hist_g, hist_b])
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def calculate_histograms(image_files: list[str]):
    bad_images = []
    histograms = []
    start_time = time.time()
    with Bar("Computing histograms...", max=len(image_files)) as bar:
        for image in image_files:
            histogram = compute_histogram(image)
            if histogram is None:
                bad_images.append(image)
            else:
                histograms.append(histogram)
            bar.next()
    histograms = np.array(histograms)

    print("took " + time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)))
    return histograms, bad_images

def get_imagefile_lists():
    real_images = get_imagefile_list(REAL_IMAGES_FOLDER)
    fake_images = get_imagefile_list(FAKE_IMAGES_FOLDER)
    return real_images + fake_images

def get_imagefile_list(folder: str):
    return [f"{folder}/{file}" 
            for file in os.listdir(folder) 
            if file.lower().endswith(('.jpg', '.png'))][:int(IMAGE_TRAIN_COUNT/2)]

def get_image_categories() -> list[str]:
    real_category_list = len(get_imagefile_list(REAL_IMAGES_FOLDER)) * ["REAL"]
    fake_category_list = len(get_imagefile_list(FAKE_IMAGES_FOLDER)) * ["FAKE"]
    category_list = real_category_list + fake_category_list

    return category_list

def load_or_calculate_histograms():
    if os.path.exists(HISTOGRAMS_FILE):
        histograms = np.load(HISTOGRAMS_FILE)
        if len(histograms) != round((IMAGE_TRAIN_COUNT  - .5) / 2) * 2 :
            os.remove(HISTOGRAMS_FILE)
            return load_or_calculate_histograms()
        return histograms
    else:
        image_files = get_imagefile_lists()
        histograms, bad_images = calculate_histograms(image_files)

        if len(bad_images) > 0: delete_bad_images(bad_images)
        np.save(HISTOGRAMS_FILE, histograms)
        return histograms

def delete_bad_images(images: list):
    for image in images:
        os.remove(image)
    print(f"\nDeleted {len(images)} bad images!")


def main():
    histograms = load_or_calculate_histograms()
    categories = get_image_categories()

    print(f"\nLoaded a total of {len(histograms)} images!")

    X_train, X_test, y_train, y_test = train_test_split(histograms, categories, test_size=TEST_SIZE)

    start_time = time.time()
    clf = model_helper.get_model(X_train, y_train, save=False)
    print("took " + time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)))

    # making predictions
    predictions = clf.predict(X_test)

    score = accuracy_score(y_test, predictions)
    print("\nAccuracy score: " + str(score))


if __name__ == "__main__":
    main()