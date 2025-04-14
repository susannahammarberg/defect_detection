import matplotlib.pyplot as plt

def plot_training_history(history):
    # Show train- and validation curves
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()