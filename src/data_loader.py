import cv2
import os
import numpy as np

def load_images(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return img


def load_data(data_dir):
    # load images. Load labels from folder-names.
    images = []
    labels = []

    print(os.listdir(data_dir))

    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            # take labels from folder-naming. Sorted with class.
            label = class_dir
            #TODO obs tem just loading 11 images
            for file_name in os.listdir(class_path)[0:11]:
                file_path = os.path.join(class_path, file_name)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

