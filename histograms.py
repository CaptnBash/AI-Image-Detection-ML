import sys
from general_processing import *
from tempfile import NamedTemporaryFile

import cv2
import numpy as np


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

def main():
    train_and_predict(REAL_HISTOGRAMS_FILE, FAKE_HISTOGRAMS_FILE, compute_histogram)


if __name__ == "__main__":
    main()