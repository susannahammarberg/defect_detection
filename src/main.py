import numpy as np
import matplotlib.pyplot as plt
import os


from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.preprocessing import convert_labels
from src.model import build_model
from src.utils import plot_training_history

def main():


    # Load train data
    print("Loading training data...")
    train_dir = r"../data/casting_data/train"
    # temporarily just loading 11 of each type of image (ok or defect)
    train_images, train_labels = load_data(train_dir)
    print(train_images.shape)
    print(train_labels)
    # load test data
    # print("Loading testing data...")
    # test_dir = r"../data/casting_data/test"
    # test_images, test_labels = load_data(test_dir)

    # temp: paste into notebook:
    "section to plot example images"
    # create a list of random numbers and plot them

    example_images_idx = np.sort(np.random.choice(21, 9, replace=False))
    print(example_images_idx)

    fig, ax = plt.subplots(3, 3, figsize=(8, 8))
    plt.suptitle('Example images')
    for i, idx in enumerate(example_images_idx):
        row, col = divmod(i, 3)
        ax[row, col].imshow(train_images[idx], cmap='gray')
        ax[row, col].set_title(train_labels[idx])
        ax[row, col].axis('off')

    plt.tight_layout()
    plt.show()


    # preprocess data
    print('preprocessing data...')
    train_images = preprocess_data(train_images)
    train_labels = convert_labels(train_labels)
    print(train_images.shape)
    print(train_labels)
    a=1

    # Build and train the model
    model = build_model()
    history = model.fit(train_images, train_labels, validation_split=0.2, epochs=10, batch_size=32)

    # Display the results
    #plot_training_history(history)

    print('Bosh!')


if __name__ == "__main__":

    main()

