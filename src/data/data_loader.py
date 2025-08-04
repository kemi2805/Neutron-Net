# src/data/data_loader_pytorch.py
"""
PyTorch data loading utilities for neutron star diffusion.
Converted from TensorFlow implementation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
import os
from pathlib import Path


class NeutronStarDataset(Dataset):
    """Dataset class for neutron star data."""
    
    def __init__(
        self,
        data: np.ndarray,
        transform: Optional[callable] = None,
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data: Numpy array of shape (N, H, W) or (N, H, W, C)
            transform: Optional transform to apply to data
            normalize: Whether to normalize data to [0, 1]
        """
        self.data = data
        self.transform = transform
        self.normalize = normalize
        
        # Ensure data is 4D: (N, C, H, W)
        if len(data.shape) == 3:
            # Add channel dimension: (N, H, W) -> (N, 1, H, W)
            self.data = data[:, None, :, :]
        elif len(data.shape) == 4 and data.shape[-1] < data.shape[1]:
            # Convert from (N, H, W, C) to (N, C, H, W)
            self.data = np.transpose(data, (0, 3, 1, 2))
        
        # Normalize to [0, 1] if requested
        if self.normalize:
            self.data = self.data.astype(np.float32)
            if self.data.max() > 1.0:
                self.data = self.data / self.data.max()
            self.data = np.clip(self.data, 0, 1)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.data[idx].astype(np.float32)
        sample = torch.from_tensor(sample)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class GaussianNoise:
    """Add Gaussian noise to data."""
    
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.std + self.mean
        return torch.clamp(x + noise, 0, 1)


class RandomHorizontalFlip:
    """Randomly flip images horizontally."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            return torch.flip(x, dims=[-1])  # Flip along width
        return x


class RandomVerticalFlip:
    """Randomly flip images vertically."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            return torch.flip(x, dims=[-2])  # Flip along height
        return x


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x


def load_neutron_star_data(
    data_path: str,
    validation_split: float = 0.1,
    batch_size: int = 8,
    num_workers: int = 4,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Load neutron star data and create train/validation dataloaders.
    
    Args:
        data_path: Path to the .npy file containing the data
        validation_split: Fraction of data to use for validation
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        augment: Whether to apply data augmentation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load data
    print(f"Loading data from {data_path}")
    data = np.load(data_path, mmap_mode='r' if os.path.getsize(data_path) > 1e9 else None)
    print(f"Loaded data with shape: {data.shape}")
    
    # Split data
    if validation_split > 0:
        train_data, val_data = train_test_split(
            data, test_size=validation_split, random_state=42
        )
    else:
        train_data = data
        val_data = data[:min(100, len(data))]  # Small validation set
    
    # Create transforms
    if augment:
        train_transform = Compose([
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
            GaussianNoise(0.0, 0.02)
        ])
    else:
        train_transform = None
    
    val_transform = None
    
    # Create datasets
    train_dataset = NeutronStarDataset(train_data, transform=train_transform)
    val_dataset = NeutronStarDataset(val_data, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    print(f"Created train loader with {len(train_dataset)} samples")
    print(f"Created validation loader with {len(val_dataset)} samples")
    
    return train_loader, val_loader


def load_and_filter_data(filepath: str) -> pd.DataFrame:
    """Load and filter data from Parquet file."""
    columns_to_read = ['eos', 'rho_c', 'M', 'R', 'J', 'I', 'r_ratio']
    data = pd.read_parquet(filepath, columns=columns_to_read)
    return data


def interpolate_r_ratio(
    df: pd.DataFrame,
    grid_size: int = 256,
    method: str = 'cubic'
) -> np.ndarray:
    """Interpolate r_ratio values onto a regular grid."""
    unique_rho_c = np.unique(df['rho_c'])
    unique_J = np.unique(df['J'])
    
    grid_size = max(len(unique_rho_c), len(unique_J))
    r_ratio_grid = np.full((grid_size, grid_size), fill_value=0)
    
    for i, rho in enumerate(unique_rho_c):
        for j, J_val in enumerate(unique_J):
            match = df[(df['rho_c'] == rho) & (df['J'] == J_val)]
            if not match.empty:
                r_ratio_grid[i, j] = match['r_ratio'].values[0]
    
    return r_ratio_grid


def process_all_eos(
    filepath: str,
    grid_size: int = 256,
    method: str = 'cubic',
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """Process dataset by filtering, interpolating, and splitting."""
    data = load_and_filter_data(filepath)
    eos_values = data['eos'].unique()
    
    grid_data = np.zeros((10, grid_size, grid_size))
    
    for i, eos in enumerate(eos_values[:10]):
        print(f"Processing EOS {eos}")
        df = data[data['eos'] == eos]
        min_r_ratio = df['r_ratio'].min()
        
        grid_data_eos = interpolate_r_ratio(df, grid_size, method)
        grid_data_eos[grid_data_eos < min_r_ratio] = 0
        grid_data[i] = grid_data_eos
    
    train_data, val_data = train_test_split(grid_data, test_size=test_size)
    return train_data, val_data
