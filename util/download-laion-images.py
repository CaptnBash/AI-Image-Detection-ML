from datasets import load_dataset
import os, hashlib, socket
from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor, as_completed
import settings

images_count = settings.IMAGE_DOWNLOAD_COUNT
folder = settings.DOWNLOADED_IMAGES_FOLDER
login = settings.LOGIN
max_workers = settings.MAX_WORKERS

socket.setdefaulttimeout(3)
# Load dataset and get URLs
dataset = load_dataset("laion/laion-high-resolution", streaming=True)
urls = [item["URL"] for item in dataset["train"].take(images_count)]
os.makedirs(folder, exist_ok=True)

# Download function with error handling
def download_image(url_idx):
    url = urls[url_idx]
    try:
        name = hashlib.sha256(url.encode()).hexdigest()
        urlretrieve(url=url, filename=f"{folder}/{name}.jpg")
        return True
    except Exception as e:
        print(f"Failed to download image {url_idx}: {url} - {str(e)}")
        return False

# Multithreaded download

download_count = 0
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all download tasks
    futures = {executor.submit(download_image, i): i for i in range(len(urls))}

    # Process completed downloads
    for future in as_completed(futures):
        if future.result():
            download_count += 1

print(f"\nFinished. Downloaded {download_count} images from a total {images_count}")