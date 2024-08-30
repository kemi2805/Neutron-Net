# utils.py
from autoencoder import AutoEncoder, AutoEncoderParams
from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union


def load_ae (
        params: AutoEncoderParams,
) -> AutoEncoder:
    # Loading the autoencoder
    print("Init AE")
    ae = AutoEncoder(params=params)
    return ae

def generate_random_data(
    image_shape: Tuple[int, int, int],
    num_samples: int,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Generates random image data with specified shape and number of samples.

    Args:
        image_shape (Tuple[int, int, int]): Shape of individual images (height, width, channels).
        num_samples (int): Number of images to generate.
        dtype (np.dtype, optional): Data type of the generated array. Defaults to np.float32.

    Returns:
        np.ndarray: Array of random image data.
    """
    return np.random.rand(*((num_samples,) + image_shape)).astype(dtype)

def load_random_data(
    image_shape: Tuple[int, int, int],
    train_size: int,
    val_size: int,
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates random training and validation data based on specified image shape and sizes.

    Args:
        image_shape (Tuple[int, int, int]): Shape of individual images (height, width, channels).
        train_size (int): Number of training samples.
        val_size (int): Number of validation samples.
        dtype (np.dtype, optional): Data type of the generated arrays. Defaults to np.float32.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the training and validation data arrays.
    """
    train_data = generate_random_data(image_shape, train_size, dtype)
    val_data = generate_random_data(image_shape, val_size, dtype)
    return train_data, val_data

def plot_original_vs_reconstructed(
    autoencoder: Model,
    data: np.ndarray,
    num_samples: int = 5,
    file_path: Union[str, None] = None
) -> None:
    """
    Plots original vs reconstructed images from the autoencoder.

    Args:
        autoencoder (Model): The autoencoder model used for reconstruction.
        data (np.ndarray): Array of images to be reconstructed, with shape (num_samples, height, width, channels).
        num_samples (int, optional): Number of samples to plot. Default is 5.

    Returns:
        None
    """
    # Ensure num_samples does not exceed the number of available images
    num_samples = min(num_samples, data.shape[0])

    # Predict reconstructed images
    try:
        reconstructed = autoencoder.predict(data[:num_samples])
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=num_samples, ncols=2, figsize=(15, 3 * num_samples))

    # Normalize pixel values for consistent visualization
    vmin = np.min([np.min(data), np.min(reconstructed)])
    vmax = np.max([np.max(data), np.max(reconstructed)])

    # Plot original and reconstructed images
    for i in range(num_samples):
        axes[i, 0].imshow(data[i], vmin=vmin, vmax=vmax)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(reconstructed[i], vmin=vmin, vmax=vmax)
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].axis('off')

    # Adjust layout so plots do not overlap
    plt.tight_layout()

    if file_path is None:
        # Set a default file path in the current working directory
        file_path = "bad_plots.png"
    
    # Save the plot as a PDF file
    plt.savefig("/mnt/rafast/miler/bad_plots.pdf")  # Save as PDF file

    # Show the plot (optional)
    plt.show()

    return None
