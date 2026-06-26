import cv2
import matplotlib.pyplot as plt
from numpy import ndarray

from general_processing import *
from settings import *


def calculate_2dft(image_file) -> ndarray:
    image_ndarray = plt.imread(image_file)

    # resizing the images to 512x512 so all arrays are of the same shape
    image_ndarray = cv2.resize(image_ndarray, (512, 512), interpolation=cv2.INTER_AREA)

    image_ndarray = image_ndarray[:, :, :3].mean(axis=2)  # Convert to grayscale

    ft = np.fft.ifftshift(image_ndarray)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)

    ft = ft.real
    # ft = ft.astype(np.float32) # optional for reducing storage size (16 for more compression)
    return ft


def calculate_2dft_flattened(image_file) -> ndarray:
    ft = calculate_2dft(image_file)
    ft = ft.flatten()
    return ft


def show_plot(nparray):
    plt.imshow(np.log(abs(nparray)))
    plt.axis("off")
    plt.show()


def main():
    train_and_predict(REAL_FT_FILE, FAKE_FT_FILE, calculate_2dft_flattened)


if __name__ == "__main__":
    main()
