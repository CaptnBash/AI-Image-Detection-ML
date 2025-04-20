from huggingface_hub import login

HISTOGRAMS_FILE = "histograms.npy"
MODEL_FILE = "model.pkl"

TEST_SIZE = 0.2 # 20% of all images are used for testing

IMAGE_DOWNLOAD_COUNT = 200000
DOWNLOADED_IMAGES_FOLDER = "laion-images" # change to whatever folder
#LOGIN = login(token="your-token") # input your huggingface token
MAX_WORKERS = 20  # Adjust based on your network bandwidth (10-50 typically good)

REAL_IMAGES_FOLDER = "REAL"
FAKE_IMAGES_FOLDER = "FAKE"