# src/utils/visualization.py
"""
Consolidated visualization module for neutron star diffusion in PyTorch.
Migrated and cleaned up from TensorFlow version.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict, Any
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import warnings

# Set style for better plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
warnings.filterwarnings('ignore', category=UserWarning)


class NeutronStarVisualizer:
    """
    Comprehensive visualization class for neutron star diffusion models.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        style: str = 'scientific'
    ):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Resolution for saved figures
            style: Plot style ('scientific', 'presentation', 'publication')
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'neutron_star': '#8c1538',
            'physics': '#9467bd'
        }
        
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style based on selected style."""
        if self.style == 'scientific':
            plt.rcParams.update({
                'font.size': 11,
                'font.family': 'serif',
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 16
            })
        elif self.style == 'presentation':
            plt.rcParams.update({
                'font.size': 14,
                'font.family': 'sans-serif',
                'axes.labelsize': 16,
                'axes.titlesize': 18,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.titlesize': 20
            })


def plot_grid_data(
    data: np.ndarray,
    title: str = "r_ratio Grid",
    xlabel: str = "J (Angular Momentum)",
    ylabel: str = r"$\rho_c$ (Central Density)",
    file_path: Optional[Union[str, Path]] = None,
    show_colorbar: bool = True,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Plot neutron star grid data with proper labels and formatting.
    
    Args:
        data: 2D or 3D array of grid data
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        file_path: Path to save the figure
        show_colorbar: Whether to show colorbar
        figsize: Figure size
    """
    # Handle different data shapes
    if len(data.shape) == 3:
        plot_data = data[0]  # Take first sample
    elif len(data.shape) == 4:
        plot_data = data[0, 0] if data.shape[1] == 1 else data[0]  # Handle channel dimension
    else:
        plot_data = data
    
    print(f"Plotting grid data with shape: {plot_data.shape}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(
        plot_data,
        cmap='viridis',
        aspect='auto',
        origin='lower',
        interpolation='bilinear'
    )
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('r_ratio', rotation=270, labelpad=20)
    
    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Grid plot saved to: {file_path}")
    
    plt.show()


def plot_original_vs_reconstructed(
    model: nn.Module,
    data: np.ndarray,
    num_samples: int = 5,
    file_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    title: str = "Original vs Reconstructed"
) -> None:
    """
    Plot original vs reconstructed images from PyTorch autoencoder.
    
    Args:
        model: PyTorch autoencoder model
        data: Input data array
        num_samples: Number of samples to plot
        file_path: Path to save the figure
        device: Device to run inference on
        title: Plot title
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Ensure num_samples doesn't exceed available data
    num_samples = min(num_samples, data.shape[0])
    
    # Prepare data for PyTorch model
    if len(data.shape) == 3:
        # Add channel dimension: (N, H, W) -> (N, 1, H, W)
        model_input = data[:num_samples, None, :, :]
    elif len(data.shape) == 4 and data.shape[-1] == 1:
        # Convert NHWC to NCHW: (N, H, W, 1) -> (N, 1, H, W)
        model_input = np.transpose(data[:num_samples], (0, 3, 1, 2))
    else:
        model_input = data[:num_samples]
    
    # Convert to tensor and run inference
    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(model_input.astype(np.float32)).to(device)
        try:
            reconstructed_tensor = model(input_tensor)
            reconstructed = reconstructed_tensor.cpu().numpy()
        except Exception as e:
            print(f"Error during model inference: {e}")
            return
    
    # Prepare data for visualization
    if len(data.shape) == 3:
        original_vis = data[:num_samples]
    elif len(data.shape) == 4:
        original_vis = data[:num_samples, :, :, 0] if data.shape[-1] == 1 else data[:num_samples, :, :, 0]
    else:
        original_vis = data[:num_samples]
    
    # Handle reconstructed data format
    if len(reconstructed.shape) == 4:
        reconstructed_vis = reconstructed[:, 0, :, :] if reconstructed.shape[1] == 1 else reconstructed[:, 0, :, :]
    else:
        reconstructed_vis = reconstructed
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Plot comparisons
    for i in range(num_samples):
        # Original
        im1 = axes[i, 0].imshow(original_vis[i], cmap='viridis', aspect='auto')
        axes[i, 0].set_title(f'Original {i+1}')
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0], shrink=0.8)
        
        # Reconstructed
        im2 = axes[i, 1].imshow(reconstructed_vis[i], cmap='viridis', aspect='auto')
        axes[i, 1].set_title(f'Reconstructed {i+1}')
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1], shrink=0.8)
    
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save if path provided
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {file_path}")
    
    plt.show()


def plot_encoder_features(
    encoder: nn.Module,
    data: np.ndarray,
    num_samples: int = 3,
    max_features: int = 16,
    file_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None
) -> None:
    """
    Plot encoder feature maps for neutron star data.
    
    Args:
        encoder: PyTorch encoder model
        data: Input data
        num_samples: Number of input samples to show
        max_features: Maximum number of feature maps to show
        file_path: Path to save the figure
        device: Device to run inference on
    """
    if device is None:
        device = next(encoder.parameters()).device
    
    num_samples = min(num_samples, data.shape[0])
    
    # Prepare input
    if len(data.shape) == 3:
        model_input = data[:num_samples, None, :, :]
    elif len(data.shape) == 4 and data.shape[-1] == 1:
        model_input = np.transpose(data[:num_samples], (0, 3, 1, 2))
    else:
        model_input = data[:num_samples]
    
    # Get encoder features
    encoder.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(model_input.astype(np.float32)).to(device)
        features = encoder(input_tensor).cpu().numpy()
    
    # Determine layout
    n_features = min(features.shape[1], max_features)
    
    fig = plt.figure(figsize=(20, 4 * num_samples))
    gs = GridSpec(num_samples, n_features + 1, figure=fig)
    
    for i in range(num_samples):
        # Original image
        ax_orig = fig.add_subplot(gs[i, 0])
        ax_orig.imshow(model_input[i, 0], cmap='viridis', aspect='auto')
        ax_orig.set_title(f'Input {i+1}')
        ax_orig.axis('off')
        
        # Feature maps
        for j in range(n_features):
            ax_feat = fig.add_subplot(gs[i, j + 1])
            feature_map = features[i, j]
            
            # Normalize feature map for visualization
            if feature_map.max() != feature_map.min():
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            
            ax_feat.imshow(feature_map, cmap='viridis', aspect='auto')
            ax_feat.set_title(f'Feature {j+1}')
            ax_feat.axis('off')
    
    plt.suptitle('Encoder Feature Maps', fontsize=16, y=0.98)
    plt.tight_layout()
    
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Feature plot saved to: {file_path}")
    
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = None,
    file_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot training history with multiple metrics.
    
    Args:
        history: Dictionary with training history
        metrics: List of metrics to plot
        file_path: Path to save the figure
        figsize: Figure size
    """
    if metrics is None:
        metrics = ['loss', 'learning_rate']
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in history and history[m]]
    n_metrics = len(available_metrics)
    
    if n_metrics == 0:
        print("No valid metrics found in history")
        return
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(figsize[0], figsize[1]))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        values = history[metric]
        epochs = range(1, len(values) + 1)
        
        axes[i].plot(epochs, values, linewidth=2, label=f'Train {metric}')
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history and history[val_metric]:
            val_values = history[val_metric]
            # Validation might be recorded less frequently
            val_epochs = np.linspace(1, len(values), len(val_values))
            axes[i].plot(val_epochs, val_values, linewidth=2, 
                        linestyle='--', label=f'Val {metric}')
        
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()} Over Time')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
        # Use log scale for learning rate
        if 'learning_rate' in metric.lower() or 'lr' in metric.lower():
            axes[i].set_yscale('log')
    
    plt.suptitle('Training History', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {file_path}")
    
    plt.show()


def plot_physics_validation_results(
    metrics: Dict[str, float],
    file_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot physics validation results as bar chart.
    
    Args:
        metrics: Dictionary of validation metrics
        file_path: Path to save the figure
        figsize: Figure size
    """
    # Filter out individual sample metrics
    plot_metrics = {k: v for k, v in metrics.items() 
                   if not k.startswith('sample_') and isinstance(v, (int, float))}
    
    if not plot_metrics:
        print("No valid metrics to plot")
        return
    
    # Prepare data
    metric_names = [k.replace('_', ' ').title() for k in plot_metrics.keys()]
    metric_values = list(plot_metrics.values())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bars
    bars = ax.bar(range(len(metric_names)), metric_values, alpha=0.8)
    
    # Color bars based on values
    for bar, value in zip(bars, metric_values):
        if value > 0.8:
            bar.set_color('#2ca02c')  # Green for good
        elif value > 0.6:
            bar.set_color('#ff7f0e')  # Orange for okay
        elif value > 0.4:
            bar.set_color('#d62728')  # Red for poor
        else:
            bar.set_color('#8c1538')  # Dark red for very poor
    
    # Formatting
    ax.set_xlabel('Validation Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Physics Validation Results')
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.set_ylim(0, max(1.1, max(metric_values) * 1.1))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Excellent (>0.8)'),
        Patch(facecolor='#ff7f0e', label='Good (0.6-0.8)'),
        Patch(facecolor='#d62728', label='Poor (0.4-0.6)'),
        Patch(facecolor='#8c1538', label='Very Poor (<0.4)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Physics validation plot saved to: {file_path}")
    
    plt.show()


def plot_data_distribution_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    file_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Compare distributions of original and reconstructed data.
    
    Args:
        original: Original data array
        reconstructed: Reconstructed data array
        file_path: Path to save the figure
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig)
    
    # Flatten data for analysis
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    # Remove zeros for better visualization
    orig_nonzero = orig_flat[orig_flat > 0]
    recon_nonzero = recon_flat[recon_flat > 0]
    
    # 1. Histogram comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(orig_nonzero, bins=50, alpha=0.7, label='Original', density=True, color='blue')
    ax1.hist(recon_nonzero, bins=50, alpha=0.7, label='Reconstructed', density=True, color='red')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Value Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q plot
    ax2 = fig.add_subplot(gs[0, 1])
    from scipy import stats
    min_len = min(len(orig_nonzero), len(recon_nonzero))
    if min_len > 100:
        # Sample for Q-Q plot if too many points
        sample_size = min(1000, min_len)
        orig_sample = np.random.choice(orig_nonzero, sample_size, replace=False)
        recon_sample = np.random.choice(recon_nonzero, sample_size, replace=False)
        
        orig_quantiles = np.sort(orig_sample)
        recon_quantiles = np.sort(recon_sample)
        
        ax2.scatter(orig_quantiles, recon_quantiles, alpha=0.6, s=2)
        ax2.plot([0, 1], [0, 1], 'r--', label='Perfect Match')
        ax2.set_xlabel('Original Quantiles')
        ax2.set_ylabel('Reconstructed Quantiles')
        ax2.set_title('Q-Q Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Statistics comparison
    ax3 = fig.add_subplot(gs[0, 2])
    stats_orig = [orig_flat.mean(), orig_flat.std(), np.median(orig_flat)]
    stats_recon = [recon_flat.mean(), recon_flat.std(), np.median(recon_flat)]
    stat_names = ['Mean', 'Std', 'Median']
    
    x = np.arange(len(stat_names))
    width = 0.35
    
    ax3.bar(x - width/2, stats_orig, width, label='Original', alpha=0.8)
    ax3.bar(x + width/2, stats_recon, width, label='Reconstructed', alpha=0.8)
    ax3.set_xlabel('Statistics')
    ax3.set_ylabel('Value')
    ax3.set_title('Statistical Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(stat_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Sample images comparison
    n_samples = min(3, original.shape[0])
    for i in range(n_samples):
        ax = fig.add_subplot(gs[1, i])
        
        # Get sample data
        if len(original.shape) == 4:
            orig_sample = original[i, 0] if original.shape[1] == 1 else original[i, :, :, 0]
            recon_sample = reconstructed[i, 0] if reconstructed.shape[1] == 1 else reconstructed[i, :, :, 0]
        else:
            orig_sample = original[i]
            recon_sample = reconstructed[i]
        
        # Create side-by-side comparison
        combined = np.concatenate([orig_sample, recon_sample], axis=1)
        
        im = ax.imshow(combined, cmap='viridis', aspect='auto')
        ax.set_title(f'Sample {i+1}: Original | Reconstructed')
        ax.axis('off')
        
        # Add separator line
        ax.axvline(x=orig_sample.shape[1] - 0.5, color='white', linewidth=2)
    
    plt.suptitle('Data Distribution Comparison', fontsize=16, y=0.95)
    plt.tight_layout()
    
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Distribution comparison plot saved to: {file_path}")
    
    plt.show()


def plot_loss_landscape_2d(
    model: nn.Module,
    data_loader,
    loss_fn,
    param_names: List[str],
    param_ranges: List[Tuple[float, float]],
    resolution: int = 20,
    file_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None
) -> None:
    """
    Plot 2D loss landscape for model parameters.
    
    Args:
        model: PyTorch model
        data_loader: Data loader for loss computation
        loss_fn: Loss function
        param_names: Names of parameters to vary
        param_ranges: Ranges for each parameter
        resolution: Grid resolution
        file_path: Path to save the figure
        device: Device to run on
    """
    if len(param_names) != 2 or len(param_ranges) != 2:
        raise ValueError("This function supports exactly 2 parameters")
    
    if device is None:
        device = next(model.parameters()).device
    
    # Create parameter grids
    p1_range = np.linspace(*param_ranges[0], resolution)
    p2_range = np.linspace(*param_ranges[1], resolution)
    P1, P2 = np.meshgrid(p1_range, p2_range)
    
    # Store original parameters
    original_state = model.state_dict().copy()
    
    # Compute loss landscape
    losses = np.zeros((resolution, resolution))
    
    model.eval()
    with torch.no_grad():
        for i, p1_val in enumerate(p1_range):
            for j, p2_val in enumerate(p2_range):
                # Update model parameters
                state_dict = model.state_dict()
                
                # This is a simplified example - you'd need to map param_names to actual parameters
                # For demonstration, assume we're modifying the first two parameters
                param_list = list(state_dict.values())
                if len(param_list) >= 2:
                    param_list[0].fill_(p1_val)
                    param_list[1].fill_(p2_val)
                
                # Compute loss
                total_loss = 0
                n_batches = 0
                for batch in data_loader:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(device)
                    else:
                        inputs = batch.to(device)
                    
                    outputs = model(inputs)
                    loss = loss_fn(outputs, inputs)
                    total_loss += loss.item()
                    n_batches += 1
                    
                    if n_batches >= 5:  # Limit for efficiency
                        break
                
                losses[j, i] = total_loss / n_batches
    
    # Restore original parameters
    model.load_state_dict(original_state)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    contour = ax.contourf(P1, P2, losses, levels=20, cmap='viridis')
    ax.contour(P1, P2, losses, levels=20, colors='white', alpha=0.5, linewidths=0.5)
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Loss')
    
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_title('Loss Landscape')
    
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Loss landscape plot saved to: {file_path}")
    
    plt.show()


# Convenience functions that match the original TensorFlow interface
def plot_original_vs_reconstructed_griddata(
    autoencoder: nn.Module,
    data: np.ndarray,
    num_samples: int = 5,
    file_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None
) -> None:
    """
    Wrapper for backward compatibility with TensorFlow version.
    """
    plot_original_vs_reconstructed(
        autoencoder, data, num_samples, file_path, device,
        title="Original vs Reconstructed Grid Data"
    )


def plot_encoder_filters(
    encoder: nn.Module,
    data: np.ndarray,
    num_samples: int = 5,
    file_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None
) -> None:
    """
    Wrapper for backward compatibility with TensorFlow version.
    """
    plot_encoder_features(encoder, data, num_samples, 16, file_path, device)


# Example usage and testing
if __name__ == "__main__":
    print("Testing PyTorch visualization utilities...")
    
    # Create dummy data
    dummy_data = np.random.rand(5, 64, 64, 1)
    grid_data = np.random.rand(64, 64)
    
    # Test grid plotting
    plot_grid_data(grid_data, title="Test Grid Data")
    
    # Test training history plotting
    dummy_history = {
        'loss': [1.0, 0.8, 0.6, 0.4, 0.3],
        'val_loss': [1.1, 0.9, 0.7, 0.5, 0.4],
        'learning_rate': [1e-3, 8e-4, 6e-4, 4e-4, 2e-4]
    }
    plot_training_history(dummy_history)
    
    # Test physics validation plotting
    dummy_metrics = {
        'r_ratio_valid_fraction': 0.95,
        'structural_similarity': 0.88,
        'energy_conservation': 0.72,
        'mass_radius_consistency': 0.65
    }
    plot_physics_validation_results(dummy_metrics)
    
    print("✅ Visualization utilities test completed!")