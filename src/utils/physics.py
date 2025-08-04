# src/utils/physics_pytorch.py
"""
Physics validation utilities for neutron star models in PyTorch.
Based on universal relations from neutron star research.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def validate_physics_constraints(
    model: nn.Module, 
    data: np.ndarray,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Validate generated neutron star data against known physics constraints.
    
    Args:
        model: Trained autoencoder model
        data: Validation data sample (numpy array)
        device: Device to run validation on
        
    Returns:
        Dictionary of physics validation metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Convert to torch tensor and move to device
    if isinstance(data, np.ndarray):
        if len(data.shape) == 3:
            data = data[:, None, :, :]  # Add channel dimension
        elif len(data.shape) == 4 and data.shape[-1] == 1:
            data = np.transpose(data, (0, 3, 1, 2))  # Convert to NCHW
        
        data_tensor = torch.from_numpy(data.astype(np.float32))
    else:
        data_tensor = data
    
    data_tensor = data_tensor.to(device)
    
    # Generate reconstructions
    with torch.no_grad():
        reconstructed = model(data_tensor)
    
    # Convert back to numpy for analysis
    original = data_tensor.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    metrics = {}
    
    # 1. Basic r_ratio constraints
    metrics['r_ratio_min'] = float(np.min(reconstructed))
    metrics['r_ratio_max'] = float(np.max(reconstructed))
    metrics['r_ratio_valid_fraction'] = float(
        np.mean((reconstructed >= 0) & (reconstructed <= 1.0))
    )
    
    # 2. Physical smoothness (gradients shouldn't be too extreme)
    grad_x = np.gradient(reconstructed, axis=-1)  # Gradient along width
    grad_y = np.gradient(reconstructed, axis=-2)  # Gradient along height
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    metrics['mean_gradient_magnitude'] = float(np.mean(gradient_magnitude))
    metrics['max_gradient_magnitude'] = float(np.max(gradient_magnitude))
    
    # 3. Conservation of features (should preserve key statistics)
    original_mean = np.mean(original)
    reconstructed_mean = np.mean(reconstructed)
    metrics['mean_conservation_error'] = float(abs(original_mean - reconstructed_mean))
    
    original_std = np.std(original)
    reconstructed_std = np.std(reconstructed)
    metrics['std_conservation_error'] = float(abs(original_std - reconstructed_std))
    
    # 4. Structural similarity
    metrics['structural_similarity'] = compute_structural_similarity(original, reconstructed)
    
    # 5. Mass-radius relation consistency
    metrics['mass_radius_consistency'] = check_mass_radius_consistency(reconstructed)
    
    # 6. Energy conservation (custom for neutron stars)
    metrics['energy_conservation'] = check_energy_conservation(original, reconstructed)
    
    return metrics


def compute_structural_similarity(
    original: np.ndarray, 
    reconstructed: np.ndarray
) -> float:
    """
    Compute structural similarity specific to neutron star data.
    
    Args:
        original: Original data array
        reconstructed: Reconstructed data array
        
    Returns:
        Structural similarity score
    """
    # Flatten arrays for correlation calculation
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    # Remove zero values for better correlation
    mask = (orig_flat != 0) & (recon_flat != 0)
    if np.sum(mask) == 0:
        return 0.0
    
    orig_nonzero = orig_flat[mask]
    recon_nonzero = recon_flat[mask]
    
    # Calculate correlation coefficient
    if len(orig_nonzero) < 2:
        return 0.0
    
    correlation = np.corrcoef(orig_nonzero, recon_nonzero)[0, 1]
    
    # Handle NaN case
    if np.isnan(correlation):
        return 0.0
    
    return float(correlation)


def check_mass_radius_consistency(data: np.ndarray) -> float:
    """
    Check if generated data respects basic mass-radius relations.
    
    For r_ratio data, higher central densities (corresponding to different regions)
    should follow physically consistent patterns.
    
    Args:
        data: Generated data array of shape (N, C, H, W)
        
    Returns:
        Consistency score between 0 and 1
    """
    consistency_scores = []
    
    for sample in data:
        if len(sample.shape) == 3:
            sample = sample[0]  # Remove channel dimension if present
        
        # Check if there's a general trend from top to bottom
        # (corresponding to different density regimes)
        top_half = sample[:sample.shape[0]//2, :]
        bottom_half = sample[sample.shape[0]//2:, :]
        
        top_half_mean = np.mean(top_half[top_half > 0])  # Only non-zero values
        bottom_half_mean = np.mean(bottom_half[bottom_half > 0])
        
        if np.isnan(top_half_mean) or np.isnan(bottom_half_mean):
            consistency_scores.append(0.5)  # Neutral score
            continue
        
        # For neutron star data, we expect certain physical trends
        # This is a simplified check - can be made more sophisticated
        if bottom_half_mean >= top_half_mean * 0.8:  # Allow some tolerance
            consistency_scores.append(1.0)
        else:
            # Calculate how much it violates the expectation
            ratio = bottom_half_mean / (top_half_mean + 1e-8)
            consistency_scores.append(max(0.0, ratio))
    
    return float(np.mean(consistency_scores))


def check_energy_conservation(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Check energy conservation between original and reconstructed data.
    
    Args:
        original: Original data array
        reconstructed: Reconstructed data array
        
    Returns:
        Energy conservation score
    """
    # Calculate total "energy" (sum of all values) for each sample
    orig_energy = np.sum(original, axis=(-2, -1))  # Sum over spatial dimensions
    recon_energy = np.sum(reconstructed, axis=(-2, -1))
    
    # Calculate relative energy difference
    energy_diff = np.abs(orig_energy - recon_energy) / (orig_energy + 1e-8)
    
    # Convert to conservation score (lower difference = higher score)
    conservation_score = np.exp(-energy_diff.mean())
    
    return float(conservation_score)


def validate_universal_relations(data: np.ndarray) -> Dict[str, float]:
    """
    Validate against universal relations from neutron star physics.
    
    Args:
        data: Generated data array
        
    Returns:
        Dictionary of universal relation metrics
    """
    metrics = {}
    
    # This is a placeholder for more sophisticated validation
    # You would need to extract physical parameters from your r_ratio data
    # based on your specific research context
    
    # Example: Check if the data follows expected scaling relations
    for i, sample in enumerate(data):
        if len(sample.shape) == 3:
            sample = sample[0]  # Remove channel dimension
        
        # Calculate some proxy measurements
        max_val = np.max(sample)
        mean_val = np.mean(sample[sample > 0])
        
        # Check if maximum and mean values are in reasonable ranges
        if 0.1 <= max_val <= 1.0 and 0.05 <= mean_val <= 0.8:
            metrics[f'sample_{i}_physical_range'] = 1.0
        else:
            metrics[f'sample_{i}_physical_range'] = 0.0
    
    # Overall compliance score
    range_scores = [v for k, v in metrics.items() if 'physical_range' in k]
    metrics['universal_relation_compliance'] = np.mean(range_scores) if range_scores else 0.0
    
    return metrics


class PhysicsAwareLoss(nn.Module):
    """
    Physics-aware loss function that incorporates neutron star constraints.
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        physics_weight: float = 0.1,
        smoothness_weight: float = 0.01,
        conservation_weight: float = 0.05
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.physics_weight = physics_weight
        self.smoothness_weight = smoothness_weight
        self.conservation_weight = conservation_weight
        
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute physics-aware loss.
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            
        Returns:
            Dictionary containing individual loss components and total loss
        """
        # Standard reconstruction loss
        reconstruction_loss = self.mse(pred, target)
        
        # Physics constraints: r_ratio should be between 0 and 1
        physics_loss = (
            torch.relu(-pred).mean() +  # Penalty for negative values
            torch.relu(pred - 1.0).mean()  # Penalty for values > 1
        )
        
        # Smoothness constraint using finite differences
        # Gradient in x direction
        grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        # Gradient in y direction  
        grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        smoothness_loss = grad_x.pow(2).mean() + grad_y.pow(2).mean()
        
        # Conservation constraint (total energy should be preserved)
        pred_sum = torch.sum(pred, dim=(-2, -1))
        target_sum = torch.sum(target, dim=(-2, -1))
        conservation_loss = torch.abs(pred_sum - target_sum).mean()
        
        # Total loss
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.physics_weight * physics_loss +
            self.smoothness_weight * smoothness_loss +
            self.conservation_weight * conservation_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'physics_loss': physics_loss,
            'smoothness_loss': smoothness_loss,
            'conservation_loss': conservation_loss
        }


def plot_physics_validation_results(metrics: Dict[str, float], save_path: str = None):
    """
    Plot physics validation results.
    
    Args:
        metrics: Dictionary of validation metrics
        save_path: Path to save the plot
    """
    # Prepare data for plotting
    metric_names = []
    metric_values = []
    
    for key, value in metrics.items():
        if not key.startswith('sample_'):  # Skip individual sample metrics
            metric_names.append(key.replace('_', ' ').title())
            metric_values.append(value)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(metric_names)), metric_values, alpha=0.7)
    
    # Color bars based on values (green for good, red for bad)
    for bar, value in zip(bars, metric_values):
        if value > 0.8:
            bar.set_color('green')
        elif value > 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax.set_xlabel('Validation Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Physics Validation Results')
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_distributions(original: np.ndarray, reconstructed: np.ndarray, save_path: str = None):
    """
    Compare distributions of original and reconstructed data.
    
    Args:
        original: Original data array
        reconstructed: Reconstructed data array
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Flatten data for distribution analysis
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    # Remove zeros for better visualization
    orig_nonzero = orig_flat[orig_flat > 0]
    recon_nonzero = recon_flat[recon_flat > 0]
    
    # 1. Histogram comparison
    axes[0, 0].hist(orig_nonzero, bins=50, alpha=0.7, label='Original', density=True)
    axes[0, 0].hist(recon_nonzero, bins=50, alpha=0.7, label='Reconstructed', density=True)
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Value Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Q-Q plot
    from scipy import stats
    orig_sorted = np.sort(orig_nonzero)
    recon_sorted = np.sort(recon_nonzero)
    
    # Match lengths for Q-Q plot
    min_len = min(len(orig_sorted), len(recon_sorted))
    orig_qq = orig_sorted[:min_len]
    recon_qq = recon_sorted[:min_len]
    
    axes[0, 1].scatter(orig_qq, recon_qq, alpha=0.5)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Perfect match')
    axes[0, 1].set_xlabel('Original Quantiles')
    axes[0, 1].set_ylabel('Reconstructed Quantiles')
    axes[0, 1].set_title('Q-Q Plot')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Sample comparison (first few samples)
    n_samples = min(4, original.shape[0])
    for i in range(n_samples):
        orig_sample = original[i, 0] if len(original.shape) == 4 else original[i]
        recon_sample = reconstructed[i, 0] if len(reconstructed.shape) == 4 else reconstructed[i]
        
        row = 1 if i < 2 else 1
        col = i % 2
        
        if i < 2:  # Only plot first 2 samples
            im = axes[row, col].imshow(
                np.concatenate([orig_sample, recon_sample], axis=1),
                cmap='viridis', aspect='auto'
            )
            axes[row, col].set_title(f'Sample {i+1}: Original (left) vs Reconstructed (right)')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Create dummy data for testing
    batch_size, channels, height, width = 4, 1, 64, 64
    
    # Generate test data with some physical structure
    original_data = np.random.rand(batch_size, channels, height, width).astype(np.float32)
    
    # Add some structure to mimic neutron star data
    for i in range(batch_size):
        # Add radial gradient
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        mask = (x - center_x)**2 + (y - center_y)**2 < (height // 3)**2
        original_data[i, 0][mask] *= 1.5
        
        # Ensure values are in [0, 1]
        original_data[i, 0] = np.clip(original_data[i, 0], 0, 1)
    
    # Create slightly noisy reconstruction
    reconstructed_data = original_data + 0.05 * np.random.randn(*original_data.shape)
    reconstructed_data = np.clip(reconstructed_data, 0, 1)
    
    # Test physics validation
    print("Testing physics validation...")
    
    # Create a dummy model that just returns the input
    class DummyModel(nn.Module):
        def forward(self, x):
            return x + 0.01 * torch.randn_like(x)
    
    model = DummyModel()
    
    # Run validation
    metrics = validate_physics_constraints(model, original_data)
    
    print("Validation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test physics-aware loss
    print("\nTesting physics-aware loss...")
    loss_fn = PhysicsAwareLoss()
    
    pred_tensor = torch.from_numpy(reconstructed_data)
    target_tensor = torch.from_numpy(original_data)
    
    loss_dict = loss_fn(pred_tensor, target_tensor)
    
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Plot results
    plot_physics_validation_results(metrics, "physics_validation_test.png")
    compare_distributions(original_data, reconstructed_data, "distribution_comparison_test.png")
    
    print("\nPhysics validation test completed!")