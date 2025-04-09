import matplotlib.pyplot as plt

def plot_training_history(history):
    # Show train- and validation curves
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Träning')
    plt.plot(history.history['val_accuracy'], label='Validering')
    plt.xlabel('Epoch')
    plt.ylabel('Noggrannhet')
    plt.legend()
    plt.show()