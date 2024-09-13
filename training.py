# training.py
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, History
from keras.models import Model

import matplotlib.pyplot as plt
from typing import Union, List, Callable, Tuple
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

def add_noise_to_inputs(inputs: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
    """
    Adds Gaussian noise to the input data.

    Args:
        inputs (np.ndarray): Input data to which noise will be added.
        noise_factor (float, optional): The factor by which to scale the noise. Defaults to 0.1.

    Returns:
        np.ndarray: The input data with added Gaussian noise.

    Details:
        - The function generates random noise from a normal distribution with mean 0 and standard deviation based on the `noise_factor`.
        - The noise is added to the original input data to create a noisy version of the inputs.
        - Noise is applied element-wise across all input samples.
    """
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=inputs.shape)
    return inputs + noise

def train_until_plateau(
        model: Model, 
        train_data: Tuple[np.ndarray, np.ndarray], 
        val_data: Tuple[np.ndarray, np.ndarray],
        patience: int = 3, 
        noise_factor: float = 0.1, 
        max_noisy_iterations: int = 7, 
        batch_size: int = 32, 
        epochs_per_round: int = 10
) -> List[History.history]:
    """
    Trains a model until validation loss plateaus, then adds noise to the input data and continues training.

    Args:
        model (tf.keras.Model): The TensorFlow/Keras model to be trained.
        train_data (Tuple[np.ndarray, np.ndarray]): Tuple containing training data and labels (train_X, train_y).
        val_data (Tuple[np.ndarray, np.ndarray]): Tuple containing validation data and labels (val_X, val_y).
        patience (int, optional): Number of epochs with no improvement on validation loss before training stops. Defaults to 3.
        noise_factor (float, optional): Scaling factor for the amount of Gaussian noise added to the input data. Defaults to 0.1.
        max_noisy_iterations (int, optional): Maximum number of training rounds with noisy data. Defaults to 5.
        batch_size (int, optional): Number of samples per gradient update during training. Defaults to 32.
        epochs_per_round (int, optional): Number of epochs to train in each iteration before checking validation loss. Defaults to 10.

    Returns:
        None: The function doesn't return any value. The model is trained iteratively, and noise is added to the input data
              when validation loss plateaus.

    Details:
        - The function trains the model on the provided `train_data` and `val_data` until the validation loss stops improving.
        - When validation loss stops improving for `patience` epochs, noise is added to the training data using a Gaussian noise model.
        - The process repeats for a maximum of `max_noisy_iterations` iterations.
        - The training process uses EarlyStopping to monitor validation loss and stop training when it plateaus.
        - A callback is used to restore the best model weights when validation loss does not improve after `patience` epochs.
        - Noise is injected using the `add_noise_to_inputs` function, which perturbs the original training data.
    """
    
    train_X, train_y = train_data
    val_X, val_y = val_data
    
    checkpoint_path = "/mnt/rafast/miler/noisy_model"
    # Use EarlyStopping to stop training if validation loss does not improve
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_delta=0.001)
    callbacks = [checkpoint_callback, reduce_lr, early_stopping]

    noisy_iteration = 0

    history_list = []
    while noisy_iteration <= max_noisy_iterations:
        print(f"\n--- Training without noise, iteration {noisy_iteration + 1} ---")

        # Train the model until validation loss no longer improves
        history = model.fit(train_X, train_y, 
                            validation_data=(val_X, val_y),
                            epochs=epochs_per_round,
                            batch_size=batch_size,
                            callbacks=callbacks,
                            verbose=2)

        history_list.append(history)

        if noisy_iteration == max_noisy_iterations:
            break

        # Add noise to the training data and repeat the training process
        print(f"Adding noise to input data and retraining... (Iteration {noisy_iteration + 1})")
        train_X = add_noise_to_inputs(train_X, noise_factor)
        
        noisy_iteration += 1
        

    print("Training completed.")
    return history_list
