import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import os


from src.data_loader import load_data
#from src.model import build_model
#from src.utils import plot_training_history

def main():



    print('Bosh!' )


    # Load train data
    print("Loading training data...")
    train_dir = r"../data/casting_data/train"
    train_images, train_labels = load_data(train_dir)

    # load test data
    # print("Loading testing data...")
    # test_dir = r"../data/casting_data/test"
    # test_images, test_labels = load_data(test_dir)

    # temp: paste into notebook:
    "section to plot example images"
    # create a list of random numbers and plot them
    example_images_idx = np.sort(random.randint(0, 55, 5))
    print(example_images_idx)

    #for idx in example_images_idx:
    fig, ax = plt.subplots(3,3)
    plt.suptitle('Example images')
    ax[0,0].imshow(train_images[0], cmap='gray')
    ax[1,0].imshow(train_images[1], cmap='gray')
    plt.axis('off')
    plt.show()

    # preprocess data
    # images = preprocess_data(images)
    #
    # # Build and train the model
    # model = build_model()
    # history = model.fit(images, labels, validation_split=0.2, epochs=10, batch_size=32)
    #
    # # Display the results
    # plot_training_history(history)


if __name__ == "__main__":

    main()

