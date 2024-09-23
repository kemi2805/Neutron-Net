from keras.models import Model

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union


def plot_grid_data(
    data: list,
    file_path: Union[str, None] = None
) -> None:
    print("train test split data.shape:", data.shape)
    plt.figure(figsize=(16, 12))
    sns.heatmap(data[0,:,:], cmap='viridis')
    plt.xlabel('J')
    plt.ylabel('rho_c')
    plt.title('r_ratio grid')
    plt.savefig("interpolated_images.pdf")

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

def plot_original_vs_reconstructed_griddata(
    autoencoder: Model,
    data: np.ndarray,
    num_samples: int = 5,
    file_path: Union[str, None] = None
) -> None:
    """
    Plots original vs reconstructed images from the autoencoder.

    Args:
        autoencoder (Model): The autoencoder model used for reconstruction.
        data (np.ndarray): Array of images to be reconstructed, with shape (num_samples, height, width).
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
    vmin = 0
    vmax = 1

    # Plot original and reconstructed images
    for i in range(num_samples):
        axes[i, 0].imshow(data[i], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(reconstructed[i], cmap='viridis', vmin=vmin, vmax=vmax)
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

def plot_encoder_filters(
    encoder: Model,
    data: np.ndarray,
    num_samples: int = 5,
    file_path: Union[str, None] = None
) -> None:
    """
    Plots the encoder output (features) for a set of input images.

    Args:
        encoder (Model): The encoder part of the autoencoder model.
        data (np.ndarray): Array of images to pass through the encoder.
        num_samples (int, optional): Number of samples to plot. Default is 5.
        file_path (Union[str, None], optional): Path to save the plot. If None, the plot will be saved as "encoder_output.png".

    Returns:
        None
    """
    # Ensure num_samples does not exceed the number of available images
    num_samples = min(num_samples, data.shape[0])

    # Get encoder output
    encoded_output = encoder.predict(data[:num_samples])

    # Assume the shape is (num_samples, 50, 50, 256)
    h, w, num_filters = encoded_output.shape[1:]
    print(num_filters)

    # Calculate the grid size for plotting filters
    # grid_size = int(np.ceil(np.sqrt(num_filters))) Keine Ahnung wieso die Wurzel hier genommen wird
    grid_size = int(num_filters)

    # Create a figure with subplots
    fig, axes = plt.subplots(num_samples, grid_size, figsize=(30, 3 * num_samples))

    # Normalize the encoded output for better visualization
    encoded_output = (encoded_output - encoded_output.min()) / (encoded_output.max() - encoded_output.min())
    print("-------------------------\nBis hier")
    for i in range(num_samples):
        for j in range(grid_size):
            ax = axes[i, j]
            if j < num_filters:
                # Plot each filter output
                ax.imshow(encoded_output[i, :, :, j], cmap='viridis')
                ax.axis('off')
            else:
                # Turn off unused subplots
                ax.axis('off')

    plt.tight_layout()

    if file_path is None:
        # Set a default file path
        file_path = "encoder_output.png"
    
    # Save the plot as a PNG file
    plt.savefig(file_path)
    
    # Show the plot (optional)
    #plt.show()

    return None
