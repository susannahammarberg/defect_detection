import numpy as np
import cv2 as cv


def preprocess_data(images):

    # Resize and normalise the images
    images = [cv.resize(img, (128, 128)) for img in images]
    images = np.array(images) / 255.0
    return images.reshape(-1, 128, 128, 1)
