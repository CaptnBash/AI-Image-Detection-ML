import numpy as np
import matplotlib.pyplot as plt


def calculate_2dft(image_file):
    image_ndarray = plt.imread(image_file)
    image_ndarray = image_ndarray[:, :, :3].mean(axis=2)  # Convert to grayscale

    ft = np.fft.ifftshift(image_ndarray)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)

    ft = ft.real
    # ft = ft.astype(np.float32) # optional for reducing storage size (16 for more compression)

    return ft



def show_plot(nparray):
    plt.imshow(np.log(abs(nparray)))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    filename = input("File path: ")
    calculated_2dft = calculate_2dft(filename)
    show_plot(calculated_2dft)