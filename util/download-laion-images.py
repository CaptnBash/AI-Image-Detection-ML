from huggingface_hub import login
from datasets import load_dataset
import os, hashlib, socket
from urllib.request import urlretrieve

folder = "laion-images" # change to whatever folder
login(token="your-token") # input your huggingface token

images_count = 200000

socket.setdefaulttimeout(3)
dataset = load_dataset("laion/laion-high-resolution", streaming=True)

urls: str = [item["URL"] for item in dataset["train"].take(images_count)]

os.makedirs(folder, exist_ok=True)

download_count = 0
for i in range(len(urls)):
    try:
        print(f"downloading image {i} from {images_count}")
        name = hashlib.sha256(urls[i].encode()).hexdigest()
        urlretrieve(url=urls[i], filename=f"{folder}/{name}.jpg") # naming all jpg does not matter
        download_count += 1
    except Exception as e:
        print(f"failed to download: {urls[i]}")

print(f"\ndownloaded {download_count} images from a total {images_count}")