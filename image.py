import os, time, sys
import model_helper
from settings import *
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import concurrent.futures as cf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def compute_histogram(image_path, bins=256):
    stderr_fd = sys.stderr.fileno()
    stderr_save = os.dup(stderr_fd)
    
    try:
        with NamedTemporaryFile(mode='w+') as tempf:
            os.dup2(tempf.fileno(), stderr_fd)
            
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hist_r = cv2.calcHist([img_rgb], [0], None, [bins], [0, 256]).flatten()
            hist_g = cv2.calcHist([img_rgb], [1], None, [bins], [0, 256]).flatten()
            hist_b = cv2.calcHist([img_rgb], [2], None, [bins], [0, 256]).flatten()
            
            tempf.seek(0)
            if tempf.read().strip():
                return None
            
            return np.concatenate([hist_r, hist_g, hist_b])
            
    except Exception:
        return None
    finally:
        os.dup2(stderr_save, stderr_fd)
        os.close(stderr_save)

def process_batch(image_paths, max_workers=None):
    histogram_list = []
    failed_images = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(compute_histogram, path): path 
            for path in image_paths
        }
        
        for future in cf.as_completed(future_to_path):
            path = future_to_path[future]
            result = future.result()
            if result is not None:
                histogram_list.append(result)
            else:
                failed_images.append(path)
        return np.array(histogram_list), failed_images

def get_imagefile_lists():
    real_images = get_imagefile_list(REAL_IMAGES_FOLDER)
    fake_images = get_imagefile_list(FAKE_IMAGES_FOLDER)
    return real_images, fake_images

def get_imagefile_list(folder: str):
    return [f"{folder}/{file}" 
            for file in os.listdir(folder) 
            if file.lower().endswith(('.jpg', '.png'))][:int(IMAGE_TRAIN_COUNT/2)]

def get_image_categories(real_amount: int, fake_amount: int) -> list[str]:
    print(f"Loaded {real_amount + fake_amount} images ({real_amount} real, {fake_amount} fake)")
    real_category_list = ["REAL"] * real_amount
    fake_category_list = ["FAKE"] * fake_amount
    return real_category_list + fake_category_list

def load_or_calculate_histograms():
    if os.path.exists(HISTOGRAMS_FILE):
        histograms_tuple = np.load(HISTOGRAMS_FILE, allow_pickle=True)
        real_histograms, fake_histograms = histograms_tuple
        total_histograms = len(real_histograms) + len(fake_histograms)

        if total_histograms != IMAGE_TRAIN_COUNT:
            os.remove(HISTOGRAMS_FILE)
            print("Invalid histogram file. Recalculating...")
        else:
            return np.concatenate((real_histograms, fake_histograms)), len(real_histograms), len(fake_histograms)

    print("Processing images...")
    start_time = time.time()

    real_images, fake_images = get_imagefile_lists()
    real_histograms, real_bad_images = process_batch(real_images)
    fake_histograms, fake_bad_images = process_batch(fake_images)

    bad_images = real_bad_images + fake_bad_images
    if bad_images:
        delete_bad_images(bad_images)

    histograms_tuple = (real_histograms, fake_histograms)
    np.save(HISTOGRAMS_FILE, histograms_tuple)

    print(f"Processed {len(real_histograms) + len(fake_histograms)} images in "
          f"{time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time))}.")

    return np.concatenate((real_histograms, fake_histograms)), len(real_histograms), len(fake_histograms)

def delete_bad_images(images: list):
    for image in images:
        os.remove(image)
    print(f"\nDeleted {len(images)} bad images!")


def main():
    histograms, real_amount, fake_amount = load_or_calculate_histograms()
    categories = get_image_categories(real_amount, fake_amount)

    X_train, X_test, y_train, y_test = train_test_split(histograms, categories, test_size=TEST_SIZE)

    clf = model_helper.get_model(X_train, y_train, save=False)

    predictions = clf.predict(X_test)

    score = accuracy_score(y_test, predictions)
    print("\nAccuracy score: " + str(score))


if __name__ == "__main__":
    main()