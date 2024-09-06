# training.py
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, History
from keras.models import Model

import matplotlib.pyplot as plt
from typing import Union, List, Callable
import numpy as np

from autoencoder import AutoEncoder, AutoEncoderParams
from utils import load_random_data, plot_original_vs_reconstructed_griddata

def train_autoencoder(
    autoencoder: Model,
    train_data: np.ndarray,
    val_data: np.ndarray,
    epochs: int = 20,
    batch_size: int = 2,
    callbacks: Callable = ModelCheckpoint
) -> History:
    """
    This will change completey to an selfwritten trainer, which changes the input data
    depending on the epoch. Like a diffusion model
    """
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

def train_for_random_data(
        params: AutoEncoderParams,
        data_shape: List[int, int, int], 
        train_size = 100, 
        val_size: int = 20,
        epochs: int = 50, 
        batch_size: int = 2,
        checkpoint_path: str = "/mnt/rafast/miler/ae_checkpoints/"
) -> None:
    """
    Trains an autoencoder model on randomly generated data.

    Args:
        params (AutoEncoderParams): Hyperparameters for configuring the AutoEncoder model.
        data_shape (List[int, int, int]): The shape of the input data, typically in the format [height, width, channels].
        train_size (int, optional): The number of training samples to generate. Defaults to 100.
        val_size (int, optional): The number of validation samples to generate. Defaults to 20.
        epochs (int, optional): The number of training epochs. Defaults to 50.
        batch_size (int, optional): The number of samples per gradient update. Defaults to 2.
        checkpoint_path (str, optional): File path to save the model checkpoints. Defaults to '/mnt/rafast/miler/ae_checkpoints/'.

    Returns:
        None: The function does not return anything. The training process is conducted, and plots are generated for 
        the training curves and original vs. reconstructed data.

    Details:
        - The function creates an autoencoder model using the provided parameters (`params`).
        - Random training and validation datasets are generated based on the `data_shape`, `train_size`, and `val_size`.
        - The autoencoder is compiled with an Adam optimizer, MeanSquaredError loss function, and MeanAbsoluteError as a metric.
        - Training is done using the provided `epochs`, `batch_size`, and a set of callbacks including model checkpointing, 
          learning rate reduction on plateau, and early stopping.
        - Training history is plotted to show the loss curves, and reconstructed vs. original data is visualized.
    """
    autoencoder = AutoEncoder(params=params)
    
    train_data, val_data = load_random_data(data_shape, train_size, val_size)
    
    # history = train_autoencoder(autoencoder, train_data, val_data, epochs=epochs, batch_size=batch_size)
    # Initialize optimizer, loss function, and metric function
    optimizer = Adam(learning_rate=0.001)
    loss_fn = MeanSquaredError()
    metric_fn = MeanAbsoluteError()

    # Compile the autoencoder
    autoencoder.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric_fn])

    # Build the model with the specified input shape
    autoencoder.build(input_shape=(batch_size, train_data.shape[1], train_data.shape[2], train_data.shape[3]))  # Adjust as needed

    # Define callbacks
    checkpoint_path = checkpoint_path
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
    
    plot_training_curves(history)
    
    plot_original_vs_reconstructed_griddata(autoencoder, val_data)
    return None