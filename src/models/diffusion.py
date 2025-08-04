# src/models/diffusion.py
"""
PyTorch implementation of the diffusion model for neutron star data.
Converted from TensorFlow implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, List, Callable, Tuple
from dataclasses import dataclass

from .autoencoder import AutoEncoder, AutoEncoderConfig


class CosineBetaScheduler:
    """Cosine beta schedule for diffusion process."""
    
    def __init__(self, num_timesteps: int, s: float = 0.008):
        self.num_timesteps = num_timesteps
        self.s = s
        self.cosine_schedule = self._generate_cosine_schedule()
    
    def _generate_cosine_schedule(self) -> torch.Tensor:
        """Generate cosine schedule for alpha values."""
        x = torch.linspace(0, 1, self.num_timesteps)
        alpha = torch.cos(((x + self.s) / (1 + self.s)) * (math.pi / 2))
        alpha = alpha ** 2
        alpha = torch.clamp(alpha, min=0, max=0.9999)
        return alpha
    
    def get_beta(self, t: int) -> float:
        """Get beta value for a given timestep."""
        alpha_0 = self.cosine_schedule[0]
        alpha_t = self.cosine_schedule[t]
        beta_t = 1 - alpha_t / alpha_0
        return beta_t.item()
    
    def get_schedule(self) -> torch.Tensor:
        """Get the full beta schedule."""
        alpha_0 = self.cosine_schedule[0]
        alpha_t = self.cosine_schedule
        beta_t = 1 - alpha_t / alpha_0
        return beta_t


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Generate random noise tensor."""
    # Calculate padded dimensions
    padded_height = 2 * math.ceil(height / 16)
    padded_width = 2 * math.ceil(width / 16)
    
    noise = torch.randn(
        num_samples, 16, padded_height, padded_width,
        device=device, generator=generator
    )
    
    return noise


def time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    """Apply time shift to tensor t based on mu and sigma."""
    return torch.exp(torch.tensor(mu)) / (
        torch.exp(torch.tensor(mu)) + (1 / t - 1) ** sigma
    )


def get_linear_function(
    x1: float = 256,
    y1: float = 0.5,
    x2: float = 4096,
    y2: float = 1.15
) -> Callable[[float], float]:
    """Generate linear interpolation function."""
    def linear_function(x: float) -> float:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m * x + b
    return linear_function


def get_cosine_function(
    x1: float = 256,
    y1: float = 0.5,
    x2: float = 4096,
    y2: float = 1.15
) -> Callable[[float], float]:
    """Generate cosine interpolation function."""
    def cosine_function(x: float) -> float:
        normalized_x = (x - x1) / (x2 - x1)
        return y1 + (y2 - y1) / 2 * (1 - math.cos(math.pi * normalized_x))
    return cosine_function


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True
) -> List[float]:
    """Generate schedule of time steps with optional shifting."""
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
    
    if shift:
        lin_function = get_linear_function(y1=base_shift, y2=max_shift)
        mu = lin_function(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    
    return timesteps.numpy().tolist()


class NeutronNet(nn.Module):
    """
    Diffusion model for neutron star data using autoencoder backbone.
    """
    
    def __init__(
        self,
        autoencoder_config: AutoEncoderConfig,
        beta_schedule: torch.Tensor,
        accumulation_steps: int = 4
    ):
        super().__init__()
        self.model = AutoEncoder(autoencoder_config)
        self.register_buffer('beta_schedule', beta_schedule)
        self.accumulation_steps = accumulation_steps
        
        # Precompute alpha values for efficiency
        alphas = 1.0 - beta_schedule
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
    
    def forward_diffusion_step(
        self,
        image: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform forward diffusion step by adding noise to image.
        
        Args:
            image: Original image tensor [B, C, H, W]
            t: Timestep tensor [B] 
            noise: Optional noise tensor, generated if None
            
        Returns:
            Tuple of (noisy_image, noise)
        """
        if noise is None:
            noise = torch.randn_like(image)
        
        # Get alpha_bar for each timestep in batch
        alpha_bar = self.alphas_cumprod[t]
        
        # Reshape for broadcasting
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        
        # Apply noise
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        
        noisy_image = sqrt_alpha_bar * image + sqrt_one_minus_alpha_bar * noise
        
        return noisy_image, noise
    
    def reverse_diffusion_step(
        self,
        noisy_image: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform reverse diffusion step to predict noise.
        
        Args:
            noisy_image: Noisy image tensor
            t: Timestep tensor
            
        Returns:
            Predicted noise
        """
        return self.model(noisy_image)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass for training."""
        return self.reverse_diffusion_step(x, t)


def denoise(
    model: nn.Module,
    img: torch.Tensor,
    timesteps: List[float],
    guidance: float = 4.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    Apply iterative denoising to an image tensor.
    
    Args:
        model: Trained diffusion model
        img: Input image tensor
        timesteps: List of timesteps for denoising
        guidance: Guidance scale
        device: Device to run on
        
    Returns:
        Denoised image tensor
    """
    if device is None:
        device = next(model.parameters()).device
    
    img = img.to(device)
    model.eval()
    
    with torch.no_grad():
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            # Create timestep tensor
            t_tensor = torch.full((img.shape[0],), t_curr, device=device)
            
            # Predict noise
            pred = model(img, t_tensor)
            
            # Update image
            img = img + (t_prev - t_curr) * pred
    
    return img


class PhysicsConstraintLoss(nn.Module):
    """Custom loss function with physics constraints."""
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        physics_weight: float = 0.1
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.physics_weight = physics_weight
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss with physics constraints."""
        # Standard reconstruction loss
        reconstruction_loss = self.mse(pred, target)
        
        # Physics constraint: r_ratio should be between 0 and 1
        physics_loss = (
            F.relu(-pred).mean() +  # Penalty for negative values
            F.relu(pred - 1.0).mean()  # Penalty for values > 1
        )
        
        # Smoothness constraint using Sobel filters
        # Create Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=pred.dtype, device=pred.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=pred.dtype, device=pred.device).unsqueeze(0).unsqueeze(0)
        
        # Apply Sobel filters
        grad_x = F.conv2d(pred, sobel_x, padding=1)
        grad_y = F.conv2d(pred, sobel_y, padding=1)
        
        smoothness_loss = (grad_x.pow(2) + grad_y.pow(2)).mean()
        
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.physics_weight * (physics_loss + 0.01 * smoothness_loss)
        )
        
        return total_loss


# Training utilities
class DiffusionTrainer:
    """Training utilities for the diffusion model."""
    
    def __init__(
        self,
        model: NeutronNet,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
    def train_step(
        self,
        batch: torch.Tensor,
        accumulation_steps: Optional[int] = None
    ) -> float:
        """Perform single training step with gradient accumulation."""
        if accumulation_steps is None:
            accumulation_steps = self.model.accumulation_steps
            
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        batch_size = batch.shape[0]
        sub_batch_size = batch_size // accumulation_steps
        
        for i in range(accumulation_steps):
            start_idx = i * sub_batch_size
            end_idx = (i + 1) * sub_batch_size
            sub_batch = batch[start_idx:end_idx].to(self.device)
            
            # Random timestep for each sample
            t = torch.randint(
                0, len(self.model.beta_schedule),
                (sub_batch.shape[0],),
                device=self.device
            )
            
            # Forward diffusion
            noisy_images, noise = self.model.forward_diffusion_step(sub_batch, t)
            
            # Predict noise
            predicted_noise = self.model.reverse_diffusion_step(noisy_images, t)
            
            # Compute loss
            loss = self.loss_fn(predicted_noise, sub_batch) / accumulation_steps
            
            # Backward pass
            loss.backward()
            total_loss += loss.item() * accumulation_steps
        
        # Update parameters
        self.optimizer.step()
        
        return total_loss
    
    @torch.no_grad()
    def validation_step(self, batch: torch.Tensor) -> float:
        """Perform validation step."""
        self.model.eval()
        batch = batch.to(self.device)
        
        # Random timestep
        t = torch.randint(
            0, len(self.model.beta_schedule),
            (batch.shape[0],),
            device=self.device
        )
        
        # Forward diffusion
        noisy_images, noise = self.model.forward_diffusion_step(batch, t)
        
        # Predict noise
        predicted_noise = self.model.reverse_diffusion_step(noisy_images, t)
        
        # Compute loss
        loss = self.loss_fn(predicted_noise, batch)
        
        return loss.item()


# Example usage
if __name__ == "__main__":
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create autoencoder config
    ae_config = AutoEncoderConfig(
        resolution=256,
        in_channels=1,
        ch=2,
        out_ch=1,
        ch_mult=[2, 4, 8, 16],
        num_res_blocks=4,
        z_channels=4,
    )
    
    # Create beta schedule
    scheduler = CosineBetaScheduler(1000)
    beta_schedule = scheduler.get_schedule()
    
    # Create model
    model = NeutronNet(ae_config, beta_schedule, accumulation_steps=4)
    model.to(device)
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = PhysicsConstraintLoss()
    
    # Create trainer
    trainer = DiffusionTrainer(model, optimizer, loss_fn, device)
    
    # Test training step
    batch = torch.randn(8, 1, 256, 256)
    loss = trainer.train_step(batch)
    print(f"Training loss: {loss:.4f}")
    
    # Test denoising
    noisy_img = torch.randn(1, 1, 256, 256, device=device)
    timesteps = get_schedule(50, 256)
    denoised_img = denoise(model, noisy_img, timesteps, device=device)
    print(f"Denoised image shape: {denoised_img.shape}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")