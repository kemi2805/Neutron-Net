# src/utils/utilities.py
"""
Consolidated utilities module for neutron star diffusion in PyTorch.
Migrated and cleaned up from TensorFlow version.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Union, Dict, Optional, List
import h5py


def load_autoencoder(
    checkpoint_path: str,
    device: torch.device,
    model_class: nn.Module = None
) -> nn.Module:
    """
    Load a PyTorch autoencoder from checkpoint.
    
    Args:
        checkpoint_path: Path to the PyTorch checkpoint
        device: Device to load the model on
        model_class: Model class to instantiate (if not in checkpoint)
        
    Returns:
        Loaded autoencoder model
    """
    print(f"Loading autoencoder from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        # Full training checkpoint
        if model_class is None:
            # Try to get model from checkpoint config
            config = checkpoint.get('config', None)
            if config is None:
                raise ValueError("Model class required when not in checkpoint")
            model = model_class(config)
        else:
            model = model_class
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        
    else:
        # Just state dict
        if model_class is None:
            raise ValueError("Model class required for state dict only")
        model = model_class
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("Autoencoder loaded successfully!")
    return model


def load_and_filter_data(filepath: str) -> pd.DataFrame:
    """
    Load and filter data from Parquet file.
    
    Args:
        filepath: Path to the Parquet file
        
    Returns:
        Filtered DataFrame with neutron star data
    """
    columns_to_read = ['eos', 'rho_c', 'M', 'R', 'J', 'I', 'r_ratio']
    
    print(f"Loading data from: {filepath}")
    data = pd.read_parquet(filepath, columns=columns_to_read)
    print(f"Loaded {len(data)} rows with columns: {list(data.columns)}")
    
    return data


def interpolate_r_ratio(
    df: pd.DataFrame,
    grid_size: int = 256,
    method: str = 'linear'  # Changed from 'cubic' for stability
) -> np.ndarray:
    """
    Interpolate r_ratio values onto a regular grid.
    
    Args:
        df: DataFrame with 'rho_c', 'J', and 'r_ratio' columns
        grid_size: Size of the output grid
        method: Interpolation method ('linear', 'nearest', 'cubic')
        
    Returns:
        2D array of interpolated r_ratio values
    """
    # Get unique values and create coordinate arrays
    unique_rho_c = np.sort(df['rho_c'].unique())
    unique_J = np.sort(df['J'].unique())
    
    # Determine actual grid size based on available data
    actual_grid_size = min(grid_size, max(len(unique_rho_c), len(unique_J)))
    
    # Method 1: Direct mapping (for regular grids)
    if len(unique_rho_c) * len(unique_J) == len(df):
        print("Using direct grid mapping")
        r_ratio_grid = np.zeros((len(unique_rho_c), len(unique_J)))
        
        for i, rho in enumerate(unique_rho_c):
            for j, J_val in enumerate(unique_J):
                match = df[(df['rho_c'] == rho) & (df['J'] == J_val)]
                if not match.empty:
                    r_ratio_grid[i, j] = match['r_ratio'].values[0]
        
        # Resize if needed
        if r_ratio_grid.shape != (grid_size, grid_size):
            from scipy.ndimage import zoom
            zoom_factors = (grid_size / r_ratio_grid.shape[0], 
                          grid_size / r_ratio_grid.shape[1])
            r_ratio_grid = zoom(r_ratio_grid, zoom_factors, order=1)
    
    else:
        # Method 2: Scipy griddata interpolation (for irregular grids)
        print("Using scipy griddata interpolation")
        
        # Create target grid
        rho_c_min, rho_c_max = df['rho_c'].min(), df['rho_c'].max()
        J_min, J_max = df['J'].min(), df['J'].max()
        
        rho_c_grid = np.linspace(rho_c_min, rho_c_max, grid_size)
        J_grid = np.linspace(J_min, J_max, grid_size)
        rho_c_mesh, J_mesh = np.meshgrid(rho_c_grid, J_grid, indexing='ij')
        
        # Prepare data for interpolation
        points = np.column_stack([df['rho_c'].values, df['J'].values])
        values = df['r_ratio'].values
        
        # Remove any NaN values
        valid_mask = ~np.isnan(values)
        points = points[valid_mask]
        values = values[valid_mask]
        
        # Interpolate
        try:
            r_ratio_grid = griddata(
                points, values, 
                (rho_c_mesh, J_mesh), 
                method=method, 
                fill_value=0.0
            )
        except Exception as e:
            print(f"Interpolation failed with {method}, falling back to nearest")
            r_ratio_grid = griddata(
                points, values,
                (rho_c_mesh, J_mesh),
                method='nearest',
                fill_value=0.0
            )
    
    # Ensure physical constraints
    r_ratio_grid = np.clip(r_ratio_grid, 0.0, 1.0)
    
    return r_ratio_grid


def process_all_eos(
    filepath: str,
    grid_size: int = 256,
    method: str = 'linear',
    test_size: float = 0.2,
    max_eos: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process dataset by filtering, interpolating, and splitting by EOS values.
    
    Args:
        filepath: Path to the dataset file
        grid_size: Size of interpolation grid
        method: Interpolation method
        test_size: Validation split fraction
        max_eos: Maximum number of EOS to process (None for all)
        
    Returns:
        Tuple of (train_data, val_data) as numpy arrays
    """
    # Load data
    data = load_and_filter_data(filepath)
    eos_values = data['eos'].unique()
    
    if max_eos is not None:
        eos_values = eos_values[:max_eos]
    
    print(f"Processing {len(eos_values)} EOS values")
    
    # Process each EOS
    grid_data_list = []
    
    for i, eos in enumerate(eos_values):
        print(f"Processing EOS {i+1}/{len(eos_values)}: {eos}")
        
        # Filter data for current EOS
        df = data[data['eos'] == eos].copy()
        
        if len(df) == 0:
            print(f"  Warning: No data for EOS {eos}")
            continue
        
        # Check data quality
        min_r_ratio = df['r_ratio'].min()
        max_r_ratio = df['r_ratio'].max()
        print(f"  r_ratio range: [{min_r_ratio:.4f}, {max_r_ratio:.4f}]")
        
        # Interpolate data onto grid
        try:
            grid_data_eos = interpolate_r_ratio(df, grid_size, method)
            
            # Apply physics constraints
            grid_data_eos[grid_data_eos < min_r_ratio] = 0
            grid_data_eos = np.clip(grid_data_eos, 0.0, 1.0)
            
            grid_data_list.append(grid_data_eos)
            print(f"  Grid shape: {grid_data_eos.shape}, non-zero elements: {np.count_nonzero(grid_data_eos)}")
            
        except Exception as e:
            print(f"  Error processing EOS {eos}: {e}")
            continue
    
    if not grid_data_list:
        raise ValueError("No valid EOS data processed")
    
    # Convert to numpy array
    grid_data = np.stack(grid_data_list, axis=0)
    print(f"Final grid data shape: {grid_data.shape}")
    
    # Split into train/validation
    train_data, val_data = train_test_split(
        grid_data, 
        test_size=test_size, 
        random_state=42
    )
    
    print(f"Train data: {train_data.shape}, Val data: {val_data.shape}")
    
    return train_data, val_data


def generate_random_data(
    image_shape: Tuple[int, int, int],
    num_samples: int,
    dtype: np.dtype = np.float32,
    physics_constrained: bool = True
) -> np.ndarray:
    """
    Generate random image data with optional physics constraints.
    
    Args:
        image_shape: Shape of individual images (height, width, channels)
        num_samples: Number of images to generate
        dtype: Data type of generated array
        physics_constrained: Apply neutron star physics constraints
        
    Returns:
        Array of random image data
    """
    data = np.random.rand(num_samples, *image_shape).astype(dtype)
    
    if physics_constrained:
        # Add radial structure typical of neutron stars
        h, w = image_shape[:2]
        center_y, center_x = h // 2, w // 2
        
        for i in range(num_samples):
            # Create radial gradient
            y, x = np.ogrid[:h, :w]
            radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_radius = np.sqrt(center_x**2 + center_y**2)
            
            # Apply radial modulation
            radial_factor = 1.0 - (radius / max_radius) * 0.5
            
            for c in range(image_shape[2] if len(image_shape) > 2 else 1):
                if len(image_shape) > 2:
                    data[i, :, :, c] *= radial_factor
                else:
                    data[i] *= radial_factor
        
        # Ensure values in [0, 1]
        data = np.clip(data, 0, 1)
    
    return data


def load_random_data(
    image_shape: Tuple[int, int, int],
    train_size: int,
    val_size: int,
    dtype: np.dtype = np.float32,
    physics_constrained: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random training and validation data.
    
    Args:
        image_shape: Shape of individual images
        train_size: Number of training samples
        val_size: Number of validation samples
        dtype: Data type
        physics_constrained: Apply physics constraints
        
    Returns:
        Tuple of (train_data, val_data)
    """
    train_data = generate_random_data(image_shape, train_size, dtype, physics_constrained)
    val_data = generate_random_data(image_shape, val_size, dtype, physics_constrained)
    
    return train_data, val_data


def convert_data_format(
    data: np.ndarray,
    source_format: str = "NHWC",
    target_format: str = "NCHW"
) -> np.ndarray:
    """
    Convert data between different tensor formats.
    
    Args:
        data: Input data array
        source_format: Source format ("NHWC", "NCHW", "HWC", "CHW")
        target_format: Target format
        
    Returns:
        Converted data array
    """
    if source_format == target_format:
        return data
    
    # Handle different conversions
    if source_format == "NHWC" and target_format == "NCHW":
        if len(data.shape) == 4:
            return np.transpose(data, (0, 3, 1, 2))
        elif len(data.shape) == 3:
            return np.transpose(data, (2, 0, 1))
    
    elif source_format == "NCHW" and target_format == "NHWC":
        if len(data.shape) == 4:
            return np.transpose(data, (0, 2, 3, 1))
        elif len(data.shape) == 3:
            return np.transpose(data, (1, 2, 0))
    
    elif source_format == "HWC" and target_format == "CHW":
        return np.transpose(data, (2, 0, 1))
    
    elif source_format == "CHW" and target_format == "HWC":
        return np.transpose(data, (1, 2, 0))
    
    else:
        raise ValueError(f"Conversion from {source_format} to {target_format} not supported")
    
    return data


def save_data_multiple_formats(
    data: np.ndarray,
    base_path: str,
    formats: List[str] = ["npy", "h5"]
) -> Dict[str, str]:
    """
    Save data in multiple formats.
    
    Args:
        data: Data to save
        base_path: Base path without extension
        formats: List of formats to save in
        
    Returns:
        Dictionary mapping format to saved file path
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    for fmt in formats:
        if fmt == "npy":
            save_path = base_path.with_suffix(".npy")
            np.save(save_path, data)
            saved_files["npy"] = str(save_path)
        
        elif fmt == "h5":
            save_path = base_path.with_suffix(".h5")
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('data', data=data, compression='gzip')
            saved_files["h5"] = str(save_path)
        
        elif fmt == "pt":
            save_path = base_path.with_suffix(".pt")
            torch.save(torch.from_numpy(data), save_path)
            saved_files["pt"] = str(save_path)
        
        else:
            print(f"Warning: Unsupported format {fmt}")
    
    return saved_files


# Tensor utilities for PyTorch
def numpy_to_torch(
    data: np.ndarray,
    device: torch.device,
    requires_grad: bool = False
) -> torch.Tensor:
    """Convert numpy array to PyTorch tensor."""
    tensor = torch.from_numpy(data.copy()).to(device)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def get_model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, any]:
    """
    Get comprehensive model summary.
    
    Args:
        model: PyTorch model
        input_shape: Input shape for the model
        
    Returns:
        Dictionary with model statistics
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'input_shape': input_shape
    }


def print_model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> None:
    """Print formatted model summary."""
    summary = get_model_summary(model, input_shape)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Input shape: {summary['input_shape']}")
    print(f"Total parameters: {summary['total_parameters']:,}")
    print(f"Trainable parameters: {summary['trainable_parameters']:,}")
    print(f"Non-trainable parameters: {summary['non_trainable_parameters']:,}")
    print(f"Model size: {summary['model_size_mb']:.2f} MB")
    print("=" * 60)


# Example usage and testing
if __name__ == "__main__":
    print("Testing PyTorch utilities...")
    
    # Test random data generation
    image_shape = (64, 64, 1)
    train_data, val_data = load_random_data(
        image_shape, 
        train_size=10, 
        val_size=5,
        physics_constrained=True
    )
    
    print(f"Generated train data: {train_data.shape}")
    print(f"Generated val data: {val_data.shape}")
    print(f"Train data range: [{train_data.min():.3f}, {train_data.max():.3f}]")
    
    # Test format conversion
    nchw_data = convert_data_format(train_data, "NHWC", "NCHW")
    print(f"Converted to NCHW: {nchw_data.shape}")
    
    # Test tensor conversion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor_data = numpy_to_torch(nchw_data, device)
    print(f"Tensor on {device}: {tensor_data.shape}")
    
    # Test saving in multiple formats
    saved_files = save_data_multiple_formats(
        train_data,
        "test_data/sample_data",
        formats=["npy", "h5", "pt"]
    )
    
    print("Saved files:")
    for fmt, path in saved_files.items():
        print(f"  {fmt}: {path}")
    
    print("PyTorch utilities test completed!")