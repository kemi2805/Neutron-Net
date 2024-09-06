# utils.py
from autoencoder import AutoEncoder, AutoEncoderParams
from keras.models import Model

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Union, Dict




def load_ae (
        params: AutoEncoderParams,
) -> AutoEncoder:
    # Loading the autoencoder
    print("Init AE")
    ae = AutoEncoder(params=params)
    return ae

def load_and_filter_data(
        filepath: str, 
) -> pd.DataFrame:
    """
    Load and filter the data from a Parquet file based on the default columns.

    Args:
        filepath (str): Path to the Parquet file.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows with the specified eos value.
    """
    # List of columns to read
    columns_to_read = ['eos', 'rho_c', 'M', 'R', 'J', 'I', 'r_ratio']

    # Read the parquet file with only the specified columns
    data = pd.read_parquet(filepath, columns=columns_to_read)
    
    return data

def interpolate_r_ratio(
        df: pd.DataFrame, 
        grid_size: int = 256, 
        method: str = 'cubic'
) -> np.ndarray:
    """
    Interpolate the 'r_ratio' values from the provided DataFrame over a specified grid.

    Args:
        df (pd.DataFrame): A DataFrame containing 'rho_c', 'J', and 'r_ratio' columns.
                           'rho_c' and 'J' are the independent variables, while 'r_ratio' 
                           is the dependent variable to be interpolated.
        grid_size (int, optional): The number of points in each dimension for the grid. 
                                   Default is 256. A higher grid_size results in a finer grid.
        method (str, optional): The interpolation method to use. Options are 'linear', 
                                'nearest', 'cubic', etc. Default is 'cubic'.

    Returns:
        np.ndarray: A 2D array of interpolated 'r_ratio' values on a grid defined by 
                    the range of 'rho_c' and 'J'. The shape of the array is (grid_size, 
                    grid_size), corresponding to the grid defined by the specified grid_size.

    Notes:
        - The function creates a grid of points based on the minimum and maximum values 
          of 'rho_c' and 'J' from the input DataFrame.
        - It uses the `griddata` function from `scipy.interpolate` to perform the interpolation.
        - Ensure that 'rho_c', 'J', and 'r_ratio' columns exist in the input DataFrame.
    """
    # Determine the unique values of rho_c and J
    unique_rho_c = np.unique(df['rho_c'])
    unique_J = np.unique(df['J'])

    # Grid size (200x200 expected, but can be less due to missing data)
    grid_size = max(len(unique_rho_c), len(unique_J))

    # Initialize a grid filled with NaN
    r_ratio_grid = np.full((grid_size, grid_size),fill_value = 0)

    # Fill the grid with r_ratio values
    for i, rho in enumerate(unique_rho_c):
        for j, J_val in enumerate(unique_J):
            # Find the corresponding r_ratio value
            match = df[(df['rho_c'] == rho) & (df['J'] == J_val)]
            if not match.empty:
                r_ratio_grid[i, j] = match['r_ratio'].values[0]


    """    # Determine the range of rho_c and J
    rho_c_min, rho_c_max = df['rho_c'].min(), df['rho_c'].max()
    J_min, J_max = df['J'].min(), df['J'].max()

    # Create evenly spaced grid points
    rho_c_grid = np.linspace(rho_c_min, rho_c_max, grid_size)
    J_grid = np.linspace(J_min, J_max, grid_size)
    rho_c_grid, J_grid = np.meshgrid(rho_c_grid, J_grid, indexing='ij')

    # Flatten the original data arrays for interpolation
    points = np.array([df['rho_c'], df['J']]).T
    values = df['r_ratio'].values
    min_values = min(values)

    # Interpolate over the new grid
    r_ratio_grid = griddata(points, values, (rho_c_grid, J_grid), method=method, fill_value=0)
    """
    return r_ratio_grid


def process_all_eos(
        filepath: str, 
        grid_size: int = 256, 
        method: str = 'cubic', 
        test_size: float = 0.2
) -> Tuple[np.array, np.array]:
    """
    Process a dataset by filtering, interpolating, and splitting based on unique 'eos' values.

    Args:
        filepath (str): The path to the dataset file.
        grid_size (int, optional): The size of the grid for interpolation. Defaults to 256.
        method (str, optional): The interpolation method to use. Defaults to 'cubic'.
        test_size (float, optional): The proportion of the dataset to include in the validation split. Defaults to 0.2.

    Returns:
        Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
            A tuple containing two lists of griddata:
            - The first list contains training griddata.
            - The second list contains validation griddata.
    """
    # Load the entire dataset
    data = load_and_filter_data(filepath)
    
    # Get unique eos values
    eos_values = data['eos'].unique()
    
    # Lists to hold train and val data for each eos
    grid_data = np.zeros((10, grid_size, grid_size))

    for i, eos in enumerate(eos_values[:10]):
        print(eos)
        # Filter data for the current eos
        df = data[data['eos'] == eos]
        min_r_ratio = df['r_ratio'].min()
        
        # Interpolate data onto a grid
        grid_data_eos = interpolate_r_ratio(df, grid_size, method)
        
        # Set values less than min_r_ratio to zero
        grid_data_eos[np.where(grid_data_eos < min_r_ratio)] = 0
        
        grid_data[i] = (grid_data_eos)
        print(grid_data.shape)

    # Split the interpolated data into training and validation sets
    train_data, val_data = train_test_split(grid_data, test_size=test_size)

    return train_data, val_data

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
    grid_size = int(np.ceil(np.sqrt(num_filters)))

    # Create a figure with subplots
    fig, axes = plt.subplots(num_samples, grid_size, figsize=(15, 3 * num_samples))

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
    plt.show()

    return None
