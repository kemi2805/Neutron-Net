# training.py
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, History
from keras.models import Model

import matplotlib.pyplot as plt
from typing import Union
import numpy as np


def train_autoencoder(
    autoencoder: Model,
    train_data: np.ndarray,
    val_data: np.ndarray,
    epochs: int = 20,
    batch_size: int = 2
) -> History:
    """
    Trains an autoencoder model with specified parameters.

    Args:
        autoencoder (Model): The autoencoder model to be trained.
        train_data (np.ndarray): Training data, used for both input and output.
        val_data (np.ndarray): Validation data, used for both input and output.
        epochs (int, optional): Number of training epochs. Default is 20.
        batch_size (int, optional): Batch size for training. Default is 2.

    Returns:
        History: The history object containing training metrics.
    """
    # Initialize optimizer, loss function, and metric function
    optimizer = Adam(learning_rate=0.001)
    loss_fn = MeanSquaredError()
    metric_fn = MeanAbsoluteError()

    # Compile the autoencoder
    autoencoder.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric_fn])

    # Build the model with the specified input shape
    autoencoder.build(input_shape=(1, 256, 256, 3))  # Adjust as needed

    # Define callbacks
    checkpoint_path = "/mnt/rafast/miler/checkpoint.tf"
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_delta=0.001)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    callbacks = [checkpoint_callback, reduce_lr, early_stop]

    # Train the autoencoder
    history = autoencoder.fit(
        x=train_data,
        y=train_data,
        epochs=epochs,
        validation_data=(val_data, val_data),
        callbacks=callbacks,
        batch_size=batch_size,
        verbose=1
    )

    return history


def plot_training_curves(history, file_path: Union[str, None] = None) -> None:
    """
    Plots the training and validation loss curves from the training history and saves the plot.

    Args:
        history: The training history object returned by the model's fit method.
                 This should contain the 'loss' and 'val_loss' attributes.
        file_path (str, optional): The path where the plot image will be saved.
                                   If not provided or None, the plot will be saved in the current working directory.
                                   Defaults to None.

    Returns:
        None
    """
    # Create a new figure with specified size
    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    # Add legend to the plot
    plt.legend()
    plt.title('Autoencoder Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Determine the file path to save the plot
    if file_path is None:
        # Set a default file path in the current working directory
        file_path = "training_curves.png"
    
    # Save the plot to the specified file path
    plt.savefig(file_path)  # Save as PNG file or any other supported format
    print(f"Plot saved to {file_path}")
    plt.show()
    plt.close()
