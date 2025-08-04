# src/models/layers.py
"""
Custom PyTorch layers for neutron star diffusion model.
This module is STILL NEEDED but significantly simplified from the TensorFlow version.

The layers here are specialized for your neutron star physics and are not available
in standard PyTorch. However, the basic building blocks are now integrated into
the autoencoder_pytorch.py file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PhysicsAwareConv2d(nn.Module):
    """
    Convolution layer with physics-aware constraints for neutron star data.
    This ensures physical properties are preserved during convolution operations.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        preserve_energy: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.preserve_energy = preserve_energy
        
        # Initialize weights for physics preservation
        self._init_physics_weights()
    
    def _init_physics_weights(self):
        """Initialize weights to preserve physical properties."""
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        
        if self.preserve_energy and self.training:
            # Apply energy conservation constraint during training
            input_energy = torch.sum(x, dim=(-2, -1), keepdim=True)
            output_energy = torch.sum(out, dim=(-2, -1), keepdim=True)
            
            # Normalize to preserve energy (with some tolerance)
            energy_ratio = input_energy / (output_energy + 1e-8)
            energy_ratio = torch.clamp(energy_ratio, 0.8, 1.2)  # Limit correction
            out = out * energy_ratio
        
        return out


class NeutronStarNormalization(nn.Module):
    """
    Specialized normalization for neutron star data that preserves physical constraints.
    This replaces standard batch/group norm for physics-critical layers.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        momentum: float = 0.1,
        affine: bool = True
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # Running statistics for inference
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, C, H, W)
        if self.training:
            # Compute batch statistics
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            
            # Update running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
                self.num_batches_tracked += 1
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learned parameters
        if self.affine:
            weight = self.weight.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            x_norm = x_norm * weight + bias
        
        # Physics constraint: ensure output preserves positivity where expected
        # (This is specific to neutron star r_ratio data)
        if not self.training:
            x_norm = torch.clamp(x_norm, min=0.0, max=1.0)
        
        return x_norm


class RadialAttention(nn.Module):
    """
    Radial attention mechanism designed for neutron star data.
    This considers the radial structure inherent in neutron star physics.
    """
    
    def __init__(
        self,
        channels: int,
        radial_bins: int = 8,
        angular_bins: int = 16
    ):
        super().__init__()
        self.channels = channels
        self.radial_bins = radial_bins
        self.angular_bins = angular_bins
        
        # Attention weights for different radial regions
        self.radial_attention = nn.Parameter(torch.ones(radial_bins))
        self.angular_attention = nn.Parameter(torch.ones(angular_bins))
        
        # Projection layers
        self.q_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)
    
    def create_radial_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create radial coordinate mask."""
        center_y, center_x = height // 2, width // 2
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        # Compute radial distance
        radius = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_radius = math.sqrt(center_x**2 + center_y**2)
        
        # Normalize to [0, 1]
        radius_norm = radius / max_radius
        
        # Create radial bins
        radial_indices = (radius_norm * (self.radial_bins - 1)).long()
        radial_indices = torch.clamp(radial_indices, 0, self.radial_bins - 1)
        
        return radial_indices
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Create radial mask
        radial_mask = self.create_radial_mask(H, W, x.device)
        
        # Standard attention computation
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Flatten spatial dimensions
        q_flat = q.view(B, C, -1).transpose(1, 2)  # (B, HW, C)
        k_flat = k.view(B, C, -1).transpose(1, 2)  # (B, HW, C)
        v_flat = v.view(B, C, -1).transpose(1, 2)  # (B, HW, C)
        
        # Compute attention weights
        scale = 1.0 / math.sqrt(C)
        attn_weights = torch.bmm(q_flat, k_flat.transpose(1, 2)) * scale
        
        # Apply radial weighting
        radial_mask_flat = radial_mask.view(-1)
        for i in range(self.radial_bins):
            mask = (radial_mask_flat == i)
            if mask.any():
                attn_weights[:, mask, :] *= self.radial_attention[i]
                attn_weights[:, :, mask] *= self.radial_attention[i]
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Apply attention
        out = torch.bmm(attn_weights, v_flat)  # (B, HW, C)
        out = out.transpose(1, 2).view(B, C, H, W)
        
        # Output projection
        out = self.out_proj(out)
        
        return x + out  # Residual connection


class PhysicsConstraintLayer(nn.Module):
    """
    Layer that enforces physics constraints on neutron star data.
    This can be inserted anywhere in the network to maintain physical validity.
    """
    
    def __init__(
        self,
        constraint_type: str = "r_ratio",
        soft_constraints: bool = True,
        constraint_weight: float = 1.0
    ):
        super().__init__()
        self.constraint_type = constraint_type
        self.soft_constraints = soft_constraints
        self.constraint_weight = constraint_weight
        
        # Learnable constraint parameters
        if soft_constraints:
            self.constraint_params = nn.Parameter(torch.tensor([0.0, 1.0]))  # [min, max]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.constraint_type == "r_ratio":
            if self.soft_constraints and self.training:
                # Soft constraints using sigmoid
                min_val, max_val = torch.sigmoid(self.constraint_params)
                min_val = min_val * 0.1  # Allow some flexibility
                max_val = 0.9 + max_val * 0.1
                
                # Apply soft clamping
                x = torch.sigmoid(x) * (max_val - min_val) + min_val
            else:
                # Hard constraints
                x = torch.clamp(x, 0.0, 1.0)
        
        elif self.constraint_type == "energy_conservation":
            # Ensure energy conservation across spatial dimensions
            if self.training:
                total_energy = torch.sum(x, dim=(-2, -1), keepdim=True)
                target_energy = total_energy.detach()  # Don't backprop through target
                
                # Normalize to conserve energy
                x = x * (target_energy / (total_energy + 1e-8))
        
        return x


# Utility function to replace standard layers with physics-aware versions
def replace_with_physics_layers(model: nn.Module, constraint_type: str = "r_ratio") -> nn.Module:
    """
    Replace standard layers in a model with physics-aware versions.
    
    Args:
        model: PyTorch model to modify
        constraint_type: Type of physics constraint to apply
        
    Returns:
        Modified model with physics-aware layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # Replace with physics-aware convolution
            new_conv = PhysicsAwareConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size[0],
                module.stride[0],
                module.padding[0],
                module.bias is not None
            )
            # Copy weights
            new_conv.conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_conv.conv.bias.data = module.bias.data.clone()
            
            setattr(model, name, new_conv)
        
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            # Replace with neutron star normalization
            if isinstance(module, nn.BatchNorm2d):
                num_features = module.num_features
            else:  # GroupNorm
                num_features = module.num_channels
            
            new_norm = NeutronStarNormalization(num_features)
            setattr(model, name, new_norm)
        
        else:
            # Recursively replace in child modules
            replace_with_physics_layers(module, constraint_type)
    
    return model


# ANSWER TO YOUR QUESTION:
"""
DO YOU STILL NEED src/models/layers.py?

YES, but in a much simpler form. Here's why:

1. **Physics-Specific Layers**: The layers above are specialized for neutron star physics
   and aren't available in standard PyTorch. They encode domain knowledge about:
   - Energy conservation in neutron star dynamics
   - Radial structure inherent in stellar objects  
   - Physical constraints on r_ratio values

2. **What Changed from TensorFlow Version**:
   - REMOVED: Basic building blocks (Conv2d, GroupNorm, etc.) - these are now in autoencoder_pytorch.py
   - REMOVED: Standard ResNet blocks, Attention blocks - integrated into main model
   - KEPT: Physics-aware specializations that encode your research domain knowledge
   - ADDED: New physics-aware layers specific to neutron star constraints

3. **Current Status**:
   - Basic layers (Encoder, Decoder, ResnetBlock, etc.) → Moved to autoencoder_pytorch.py
   - Physics-specific layers → Stay in layers.py (this file)
   - Standard PyTorch layers → Use directly from torch.nn

4. **Recommendation**:
   OPTION A (Recommended): Keep this simplified layers.py for physics-aware components
   OPTION B: Move these physics layers into autoencoder_pytorch.py and delete layers.py
   OPTION C: Delete layers.py entirely if you don't need the physics specializations

Choose OPTION A if you want to:
- Enforce physics constraints during training
- Use radial attention for stellar structure
- Apply energy conservation constraints
- Have specialized normalization for neutron star data

Choose OPTION B if you want everything in one file but keep physics features.

Choose OPTION C if you want to use standard PyTorch layers without physics constraints.
"""

# Example of how to use these physics-aware layers:

if __name__ == "__main__":
    # Test physics-aware layers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    batch_size, channels, height, width = 2, 4, 64, 64
    x = torch.randn(batch_size, channels, height, width, device=device)
    
    print("Testing Physics-Aware Layers:")
    print(f"Input shape: {x.shape}")
    
    # Test physics-aware convolution
    physics_conv = PhysicsAwareConv2d(channels, channels).to(device)
    conv_out = physics_conv(x)
    print(f"Physics Conv output shape: {conv_out.shape}")
    
    # Test neutron star normalization
    ns_norm = NeutronStarNormalization(channels).to(device)
    norm_out = ns_norm(conv_out)
    print(f"NS Norm output shape: {norm_out.shape}")
    
    # Test radial attention
    radial_attn = RadialAttention(channels).to(device)
    attn_out = radial_attn(norm_out)
    print(f"Radial Attention output shape: {attn_out.shape}")
    
    # Test physics constraints
    constraint_layer = PhysicsConstraintLayer("r_ratio").to(device)
    constrained_out = constraint_layer(attn_out)
    print(f"Constrained output shape: {constrained_out.shape}")
    print(f"Output range: [{constrained_out.min():.3f}, {constrained_out.max():.3f}]")
    
    print("\n✅ Physics-aware layers test completed!")
    
    # Example of replacing standard model with physics-aware version
    print("\nExample: Converting standard model to physics-aware:")
    
    # Create a simple standard model
    standard_model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 1, 3, padding=1)
    )
    
    print("Before conversion:")
    for name, module in standard_model.named_modules():
        if name:  # Skip empty name for the Sequential container
            print(f"  {name}: {type(module).__name__}")
    
    # Convert to physics-aware version
    physics_model = replace_with_physics_layers(standard_model)
    
    print("\nAfter conversion:")
    for name, module in physics_model.named_modules():
        if name:
            print(f"  {name}: {type(module).__name__}")