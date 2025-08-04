"""
Physics validation for neutron star models.
Based on universal relations from your research paper.
"""
import numpy as np
import tensorflow as tf
from typing import Dict, Any


def validate_physics_constraints(model: tf.keras.Model, 
                                 data: np.ndarray) -> Dict[str, float]:
    """
    Validate generated neutron star data against known physics constraints.
    
    Args:
        model: Trained autoencoder model
        data: Validation data sample
        
    Returns:
        Dictionary of physics validation metrics
    """
    # Generate reconstructions
    reconstructed = model(data, training=False).numpy()
    
    metrics = {}
    
    # 1. Basic r_ratio constraints
    metrics['r_ratio_min'] = float(np.min(reconstructed))
    metrics['r_ratio_max'] = float(np.max(reconstructed))
    metrics['r_ratio_valid_fraction'] = float(
        np.mean((reconstructed >= 0) & (reconstructed <= 1.0))
    )
    
    # 2. Physical smoothness (gradients shouldn't be too extreme)
    grad_x = np.gradient(reconstructed, axis=1)
    grad_y = np.gradient(reconstructed, axis=2)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    metrics['mean_gradient_magnitude'] = float(np.mean(gradient_magnitude))
    metrics['max_gradient_magnitude'] = float(np.max(gradient_magnitude))
    
    # 3. Conservation of features (should preserve key statistics)
    original_mean = np.mean(data)
    reconstructed_mean = np.mean(reconstructed)
    metrics['mean_conservation_error'] = float(abs(original_mean - reconstructed_mean))
    
    original_std = np.std(data)
    reconstructed_std = np.std(reconstructed)
    metrics['std_conservation_error'] = float(abs(original_std - reconstructed_std))
    
    # 4. Structural similarity (for neutron star EOS data)
    metrics['structural_similarity'] = compute_structural_similarity(data, reconstructed)
    
    # 5. Mass-radius relation consistency (simplified check)
    metrics['mass_radius_consistency'] = check_mass_radius_consistency(reconstructed)
    
    return metrics


def compute_structural_similarity(original: np.ndarray, 
                                  reconstructed: np.ndarray) -> float:
    """
    Compute structural similarity specific to neutron star data.
    """
    # Simple structural similarity based on correlation
    flattened_orig = original.flatten()
    flattened_recon = reconstructed.flatten()
    
    correlation = np.corrcoef(flattened_orig, flattened_recon)[0, 1]
    return float(correlation)


def check_mass_radius_consistency(data: np.ndarray) -> float:
    """
    Check if generated data respects basic mass-radius relations.
    This is a simplified check - you can make it more sophisticated.
    """
    # For r_ratio data, higher central densities (bottom of image) 
    # should generally have higher r_ratio values
    
    consistency_scores = []
    
    for sample in data:
        # Check if there's a general trend from top to bottom
        top_half_mean = np.mean(sample[:sample.shape[0]//2, :])
        bottom_half_mean = np.mean(sample[sample.shape[0]//2:, :])
        
        # Bottom should generally have higher values (higher density regions)
        if bottom_half_mean >= top_half_mean:
            consistency_scores.append(1.0)
        else:
            # Calculate how much it violates the expectation
            ratio = bottom_half_mean / (top_half_mean + 1e-8)
            consistency_scores.append(max(0.0, ratio))
    
    return float(np.mean(consistency_scores))


def validate_universal_relations(data: np.ndarray) -> Dict[str, float]:
    """
    Validate against universal relations from your paper:
    - Mass ratio R = M_max / M_TOV ≈ 1.255
    - Surface oblateness relations
    """
    # This is a placeholder for more sophisticated validation
    # You would need to extract physical parameters from your r_ratio data
    
    metrics = {}
    
    # Placeholder - implement based on your specific data format
    metrics['universal_relation_compliance'] = 1.0  # Placeholder
    
    return metrics


class PhysicsConstraintLoss(tf.keras.losses.Loss):
    """Custom loss function that includes physics constraints."""
    
    def __init__(self, reconstruction_weight: float = 1.0, 
                 physics_weight: float = 0.1, name="physics_constraint_loss"):
        super().__init__(name=name)
        self.reconstruction_weight = reconstruction_weight
        self.physics_weight = physics_weight
        self.mse = tf.keras.losses.MeanSquaredError()
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute loss with physics constraints."""
        # Standard reconstruction loss
        reconstruction_loss = self.mse(y_true, y_pred)
        
        # Physics constraint: r_ratio should be between 0 and 1
        physics_loss = tf.reduce_mean(
            tf.nn.relu(-y_pred) +  # Penalty for negative values
            tf.nn.relu(y_pred - 1.0)  # Penalty for values > 1
        )
        
        # Smoothness constraint
        grad_x = tf.image.sobel_edges(y_pred)[..., 0]
        grad_y = tf.image.sobel_edges(y_pred)[..., 1]
        smoothness_loss = tf.reduce_mean(tf.square(grad_x) + tf.square(grad_y))
        
        total_loss = (self.reconstruction_weight * reconstruction_loss + 
                     self.physics_weight * (physics_loss + 0.01 * smoothness_loss))
        
        return total_loss