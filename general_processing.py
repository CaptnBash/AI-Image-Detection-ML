import os, time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from settings import *
import model_helper

from tqdm import tqdm
import numpy as np
import concurrent.futures as cf


def process_batch(image_paths, processing_function, max_workers=MAX_WORKERS):
    processed_data_list = []
    failed_images = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(processing_function, path): path
            for path in image_paths
        }
        
        for future in tqdm(cf.as_completed(future_to_path), total=len(future_to_path)):
            path = future_to_path[future]
            result = future.result()
            if result is not None:
                processed_data_list.append(result)
            else:
                failed_images.append(path)
        return np.array(processed_data_list), failed_images

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

def numpy_files_exist(real_file, fake_file):
    return os.path.exists(real_file) and os.path.exists(fake_file)

def numpy_files_valid(real_file, fake_file):
    if numpy_files_exist(real_file, fake_file):
        return os.path.getsize(real_file) > 0 and os.path.getsize(fake_file) > 0
    return False

def train_count_exceeds_max():
    image_count = len(get_imagefile_list(REAL_IMAGES_FOLDER)) + len(get_imagefile_list(FAKE_IMAGES_FOLDER))
    return IMAGE_TRAIN_COUNT > image_count

def load_or_calculate_processed_images_data(real_data_file, fake_data_file, processing_function):
    if numpy_files_valid(real_data_file, fake_data_file):
        real_data = np.load(real_data_file)
        fake_data = np.load(fake_data_file)
        total_data = len(real_data) + len(fake_data)

        if total_data != IMAGE_TRAIN_COUNT and not train_count_exceeds_max():
            print(f"Loaded {total_data} datasets, but expected {IMAGE_TRAIN_COUNT}.")
            os.remove(real_data_file)
            os.remove(fake_data_file)
            print("Invalid data file. Recalculating...")
        else:
            return np.concatenate((real_data, fake_data)), len(real_data), len(fake_data)

    print("Processing images...")
    start_time = time.time()

    real_images, fake_images = get_imagefile_lists()
    real_data, real_bad_images = process_batch(real_images, processing_function)
    fake_data, fake_bad_images = process_batch(fake_images, processing_function)

    bad_images = real_bad_images + fake_bad_images
    if bad_images:
        delete_bad_images(bad_images)

    np.save(real_data_file, real_data)
    np.save(fake_data_file, fake_data)

    print(f"Processed {len(real_data) + len(fake_data)} images in "
          f"{time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time))}.")

    return np.concatenate((real_data, fake_data)), len(real_data), len(fake_data)

def delete_bad_images(images: list):
    for image in images:
        os.remove(image)
    print(f"\nDeleted {len(images)} bad images!")

def train_and_predict(real_data_file, fake_data_file, processing_function):
    data, real_amount, fake_amount = load_or_calculate_processed_images_data(real_data_file, fake_data_file, processing_function)
    categories = get_image_categories(real_amount, fake_amount)

    X_train, X_test, y_train, y_test = train_test_split(data, categories, test_size=TEST_SIZE)

    clf = model_helper.get_model(X_train, y_train, save=False)

    predictions = clf.predict(X_test)

    score = accuracy_score(y_test, predictions)
    print("\nAccuracy score: " + str(score))


