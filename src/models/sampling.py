# src/models/sampling.py
"""
PyTorch sampling module for neutron star diffusion.
Converted from TensorFlow implementation.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Optional, Union, Tuple
from tqdm.auto import tqdm


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    channels: int = 16,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Generate a tensor of random noise with a normal distribution.

    Args:
        num_samples: Number of samples (batch size)
        height: Target height of the final image
        width: Target width of the final image
        channels: Number of channels for the noise tensor
        device: Device to create tensor on
        dtype: Data type of the tensor
        generator: Random generator for reproducibility

    Returns:
        Tensor of random noise with shape [num_samples, channels, padded_height, padded_width]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Calculate padded dimensions (align to 16-pixel boundaries)
    padded_height = 2 * math.ceil(height / 16)
    padded_width = 2 * math.ceil(width / 16)

    # Create random normal noise
    noise = torch.randn(
        num_samples, channels, padded_height, padded_width,
        dtype=dtype,
        device=device,
        generator=generator
    )

    return noise


def time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    """
    Apply time shift to tensor t based on mu and sigma parameters.

    Args:
        mu: Shift parameter
        sigma: Shape control parameter  
        t: Tensor of time steps

    Returns:
        Tensor with time steps shifted according to mu and sigma
    """
    # Convert to tensors for computation
    mu_tensor = torch.tensor(mu, device=t.device, dtype=t.dtype)
    
    # Apply the time shift formula
    numerator = torch.exp(mu_tensor)
    denominator = torch.exp(mu_tensor) + torch.pow(1.0 / t - 1.0, sigma)
    
    return numerator / denominator


def get_linear_function(
    x1: float = 256, 
    y1: float = 0.5, 
    x2: float = 4096, 
    y2: float = 1.15
) -> Callable[[float], float]:
    """
    Generate a linear function that maps x to y based on two points.

    Args:
        x1: x-coordinate of the first point
        y1: y-coordinate of the first point
        x2: x-coordinate of the second point
        y2: y-coordinate of the second point

    Returns:
        Function that computes linear interpolation
    """
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
    """
    Generate a cosine interpolation function that maps x to y based on two points.

    Args:
        x1: x-coordinate of the first point
        y1: y-coordinate of the first point
        x2: x-coordinate of the second point
        y2: y-coordinate of the second point

    Returns:
        Function that computes cosine interpolation
    """
    def cosine_function(x: float) -> float:
        # Normalize x between x1 and x2
        normalized_x = (x - x1) / (x2 - x1)
        # Apply cosine interpolation
        return y1 + (y2 - y1) / 2 * (1 - math.cos(math.pi * normalized_x))
    
    return cosine_function


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
    device: torch.device = None
) -> List[float]:
    """
    Generate a schedule of time steps with optional shifting.

    Args:
        num_steps: Number of time steps to generate
        image_seq_len: Length of the image sequence (used for shifting)
        base_shift: Base shift value
        max_shift: Maximum shift value
        shift: Whether to apply shifting to the time steps
        device: Device for tensor operations

    Returns:
        List of time steps, potentially shifted
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate time steps from 1.0 to 0.0
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    
    if shift:
        # Get the linear function for shifting
        lin_function = get_linear_function(y1=base_shift, y2=max_shift)
        mu = lin_function(image_seq_len)
        
        # Apply time shifting
        timesteps = time_shift(mu, 1.0, timesteps)
    
    # Convert to list and move to CPU
    return timesteps.cpu().numpy().tolist()


def denoise(
    model: nn.Module,
    img: torch.Tensor,
    timesteps: List[float],
    img_ids: Optional[torch.Tensor] = None,
    vec: Optional[torch.Tensor] = None,
    guidance: float = 4.0,
    show_progress: bool = False,
    device: torch.device = None
) -> torch.Tensor:
    """
    Apply iterative denoising to an image tensor using a model and guidance.

    Args:
        model: PyTorch model that predicts noise
        img: Input image tensor to denoise
        timesteps: List of time steps for the denoising process
        img_ids: Optional image identifiers or associated data
        vec: Optional additional vector input (latent vector or embedding)
        guidance: Scalar controlling the influence of guidance during denoising
        show_progress: Whether to show progress bar
        device: Device to run inference on

    Returns:
        Denoised image tensor
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Move tensors to device
    img = img.to(device)
    if img_ids is not None:
        img_ids = img_ids.to(device)
    if vec is not None:
        vec = vec.to(device)
    
    model.eval()
    
    # Create guidance vector
    batch_size = img.shape[0]
    guidance_vec = torch.full((batch_size,), guidance, device=device, dtype=img.dtype)
    
    # Iterate over time steps
    timestep_pairs = list(zip(timesteps[:-1], timesteps[1:]))
    
    if show_progress:
        timestep_pairs = tqdm(timestep_pairs, desc="Denoising")
    
    with torch.no_grad():
        for t_curr, t_prev in timestep_pairs:
            # Create time step tensor
            t_vec = torch.full((batch_size,), t_curr, device=device, dtype=img.dtype)
            
            # Prepare model inputs - adapt based on your model's interface
            model_inputs = {'img': img, 'timesteps': t_vec}
            
            if img_ids is not None:
                model_inputs['img_ids'] = img_ids
            if vec is not None:
                model_inputs['y'] = vec
            if guidance != 1.0:
                model_inputs['guidance'] = guidance_vec
            
            # Predict noise using the model
            try:
                # Try the full interface first
                pred = model(**model_inputs)
            except TypeError:
                # Fallback to simpler interface
                try:
                    pred = model(img, t_vec)
                except TypeError:
                    # Most basic interface
                    pred = model(img)
            
            # Update the image tensor
            step_size = t_prev - t_curr
            img = img + step_size * pred

    return img


def denoise_with_classifier_free_guidance(
    model: nn.Module,
    img: torch.Tensor,
    timesteps: List[float],
    conditioning: Optional[torch.Tensor] = None,
    guidance_scale: float = 7.5,
    unconditional_conditioning: Optional[torch.Tensor] = None,
    show_progress: bool = False,
    device: torch.device = None
) -> torch.Tensor:
    """
    Apply denoising with classifier-free guidance.

    Args:
        model: PyTorch diffusion model
        img: Input noisy image tensor
        timesteps: List of denoising timesteps
        conditioning: Conditional input (e.g., text embeddings)
        guidance_scale: Scale for classifier-free guidance
        unconditional_conditioning: Unconditional input for guidance
        show_progress: Whether to show progress bar
        device: Device to run on

    Returns:
        Denoised image tensor
    """
    if device is None:
        device = next(model.parameters()).device
    
    img = img.to(device)
    if conditioning is not None:
        conditioning = conditioning.to(device)
    if unconditional_conditioning is not None:
        unconditional_conditioning = unconditional_conditioning.to(device)
    
    model.eval()
    batch_size = img.shape[0]
    
    timestep_pairs = list(zip(timesteps[:-1], timesteps[1:]))
    if show_progress:
        timestep_pairs = tqdm(timestep_pairs, desc="CFG Denoising")
    
    with torch.no_grad():
        for t_curr, t_prev in timestep_pairs:
            t_vec = torch.full((batch_size,), t_curr, device=device, dtype=img.dtype)
            
            if guidance_scale != 1.0 and conditioning is not None:
                # Classifier-free guidance
                # Concatenate conditional and unconditional inputs
                img_input = torch.cat([img, img], dim=0)
                t_input = torch.cat([t_vec, t_vec], dim=0)
                
                if unconditional_conditioning is not None:
                    cond_input = torch.cat([conditioning, unconditional_conditioning], dim=0)
                else:
                    # Use zeros for unconditional
                    uncond = torch.zeros_like(conditioning)
                    cond_input = torch.cat([conditioning, uncond], dim=0)
                
                # Get predictions
                pred = model(img_input, t_input, cond_input)
                
                # Split predictions
                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                
                # Apply guidance
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                # Standard prediction
                if conditioning is not None:
                    pred = model(img, t_vec, conditioning)
                else:
                    pred = model(img, t_vec)
            
            # Update image
            step_size = t_prev - t_curr
            img = img + step_size * pred

    return img


def sample_from_model(
    model: nn.Module,
    shape: Tuple[int, int, int, int],
    timesteps: List[float],
    conditioning: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
    device: torch.device = None,
    show_progress: bool = True
) -> torch.Tensor:
    """
    Generate samples from the diffusion model starting from pure noise.

    Args:
        model: Trained diffusion model
        shape: Shape of samples to generate (batch_size, channels, height, width)
        timesteps: Denoising timesteps schedule
        conditioning: Optional conditioning input
        guidance_scale: Scale for classifier-free guidance
        generator: Random generator for reproducibility
        device: Device to run on
        show_progress: Whether to show progress bar

    Returns:
        Generated samples tensor
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Generate initial noise
    noise = torch.randn(*shape, device=device, generator=generator)
    
    # Apply denoising
    if guidance_scale != 1.0 and conditioning is not None:
        samples = denoise_with_classifier_free_guidance(
            model=model,
            img=noise,
            timesteps=timesteps,
            conditioning=conditioning,
            guidance_scale=guidance_scale,
            show_progress=show_progress,
            device=device
        )
    else:
        samples = denoise(
            model=model,
            img=noise,
            timesteps=timesteps,
            vec=conditioning,
            guidance=guidance_scale,
            show_progress=show_progress,
            device=device
        )
    
    return samples


def unpack(
    x: torch.Tensor, 
    height: int, 
    width: int,
    patch_size: int = 16
) -> torch.Tensor:
    """
    Unpack and reshape a tensor based on height and width parameters.
    
    This function is designed for models that use patch-based representations,
    like Vision Transformers or patch-based diffusion models.

    Args:
        x: Input tensor of shape [batch_size, seq_len, channels*patch_size^2]
        height: Target height for the unpacked tensor
        width: Target width for the unpacked tensor  
        patch_size: Size of each patch (default: 16)

    Returns:
        Unpacked tensor of shape [batch_size, channels, height, width]
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    
    # Calculate number of channels
    channels = x.shape[-1] // (patch_size * patch_size)
    
    # Calculate grid dimensions
    grid_h = height // patch_size
    grid_w = width // patch_size
    
    # Verify sequence length matches grid
    expected_seq_len = grid_h * grid_w
    if seq_len != expected_seq_len:
        raise ValueError(f"Sequence length {seq_len} doesn't match expected {expected_seq_len} "
                        f"for {height}x{width} image with {patch_size}x{patch_size} patches")
    
    # Reshape to patch format: [batch, grid_h, grid_w, patch_size, patch_size, channels]
    x = x.view(batch_size, grid_h, grid_w, patch_size, patch_size, channels)
    
    # Rearrange to image format: [batch, channels, height, width]
    x = x.permute(0, 5, 1, 3, 2, 4)  # [batch, channels, grid_h, patch_size, grid_w, patch_size]
    x = x.contiguous().view(batch_size, channels, height, width)
    
    return x


def pack(
    x: torch.Tensor,
    patch_size: int = 16
) -> torch.Tensor:
    """
    Pack an image tensor into patches (inverse of unpack).

    Args:
        x: Input tensor of shape [batch_size, channels, height, width]
        patch_size: Size of each patch

    Returns:
        Packed tensor of shape [batch_size, num_patches, channels*patch_size^2]
    """
    batch_size, channels, height, width = x.shape
    
    # Check if height and width are divisible by patch_size
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Image dimensions ({height}, {width}) must be divisible by patch_size ({patch_size})")
    
    grid_h = height // patch_size
    grid_w = width // patch_size
    
    # Reshape to patches: [batch, channels, grid_h, patch_size, grid_w, patch_size]
    x = x.view(batch_size, channels, grid_h, patch_size, grid_w, patch_size)
    
    # Rearrange: [batch, grid_h, grid_w, patch_size, patch_size, channels]
    x = x.permute(0, 2, 4, 3, 5, 1)
    
    # Flatten patches: [batch, num_patches, channels*patch_size^2]
    num_patches = grid_h * grid_w
    patch_dim = channels * patch_size * patch_size
    x = x.contiguous().view(batch_size, num_patches, patch_dim)
    
    return x


def interpolate_timesteps(
    timesteps: List[float],
    num_inference_steps: int,
    method: str = "linear"
) -> List[float]:
    """
    Interpolate timesteps for different inference step counts.

    Args:
        timesteps: Original timestep schedule
        num_inference_steps: Desired number of inference steps
        method: Interpolation method ("linear", "quadratic")

    Returns:
        Interpolated timestep schedule
    """
    if num_inference_steps >= len(timesteps):
        return timesteps
    
    if method == "linear":
        indices = np.linspace(0, len(timesteps) - 1, num_inference_steps)
        indices = np.round(indices).astype(int)
        return [timesteps[i] for i in indices]
    
    elif method == "quadratic":
        # Quadratic spacing for more steps near the end
        x = np.linspace(0, 1, num_inference_steps)
        x_quad = x ** 2
        indices = (x_quad * (len(timesteps) - 1)).round().astype(int)
        return [timesteps[i] for i in indices]
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


class DDIM:
    """
    Denoising Diffusion Implicit Models (DDIM) sampling.
    Provides faster sampling with fewer steps.
    """
    
    def __init__(self, eta: float = 0.0):
        """
        Initialize DDIM sampler.
        
        Args:
            eta: Controls stochasticity (0.0 = deterministic, 1.0 = DDPM)
        """
        self.eta = eta
    
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int, int],
        timesteps: List[float],
        conditioning: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        device: torch.device = None,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Generate samples using DDIM sampling.
        
        Args:
            model: Trained diffusion model
            shape: Shape of samples to generate
            timesteps: Timestep schedule
            conditioning: Optional conditioning
            generator: Random generator
            device: Device to run on
            show_progress: Show progress bar
            
        Returns:
            Generated samples
        """
        if device is None:
            device = next(model.parameters()).device
        
        # Start with noise
        x = torch.randn(*shape, device=device, generator=generator)
        
        model.eval()
        timestep_pairs = list(zip(timesteps[:-1], timesteps[1:]))
        
        if show_progress:
            timestep_pairs = tqdm(timestep_pairs, desc="DDIM Sampling")
        
        with torch.no_grad():
            for i, (t_curr, t_next) in enumerate(timestep_pairs):
                batch_size = x.shape[0]
                t_vec = torch.full((batch_size,), t_curr, device=device)
                
                # Predict noise
                if conditioning is not None:
                    noise_pred = model(x, t_vec, conditioning)
                else:
                    noise_pred = model(x, t_vec)
                
                # DDIM update step
                alpha_t = 1.0 - t_curr
                alpha_t_next = 1.0 - t_next
                
                # Predicted x0
                x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                
                # Direction to xt
                dir_xt = torch.sqrt(1 - alpha_t_next) * noise_pred
                
                # Add noise if eta > 0
                if self.eta > 0:
                    noise = torch.randn_like(x, generator=generator)
                    sigma_t = self.eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_next)
                    dir_xt = dir_xt + sigma_t * noise
                
                # Update x
                x = torch.sqrt(alpha_t_next) * x0_pred + dir_xt
        
        return x


# Example usage and testing
if __name__ == "__main__":
    print("Testing PyTorch sampling utilities...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test noise generation
    noise = get_noise(2, 256, 256, device=device)
    print(f"Generated noise shape: {noise.shape}")
    
    # Test schedule generation
    schedule = get_schedule(50, 256, device=device)
    print(f"Generated schedule with {len(schedule)} steps")
    print(f"Schedule range: [{schedule[0]:.3f}, {schedule[-1]:.3f}]")
    
    # Test interpolation functions
    linear_fn = get_linear_function()
    cosine_fn = get_cosine_function()
    print(f"Linear interpolation at 1000: {linear_fn(1000):.3f}")
    print(f"Cosine interpolation at 1000: {cosine_fn(1000):.3f}")
    
    # Test pack/unpack
    test_img = torch.randn(2, 3, 64, 64, device=device)
    packed = pack(test_img, patch_size=16)
    unpacked = unpack(packed, 64, 64, patch_size=16)
    print(f"Pack/unpack test - Original: {test_img.shape}, Packed: {packed.shape}, Unpacked: {unpacked.shape}")
    print(f"Pack/unpack matches: {torch.allclose(test_img, unpacked, atol=1e-6)}")
    
    # Test DDIM sampler
    ddim = DDIM(eta=0.0)
    print(f"DDIM sampler created with eta={ddim.eta}")
    
    # Test timestep interpolation
    original_timesteps = list(np.linspace(1.0, 0.0, 100))
    interpolated = interpolate_timesteps(original_timesteps, 20, method="linear")
    print(f"Interpolated timesteps: {len(original_timesteps)} -> {len(interpolated)}")
    
    print("✅ PyTorch sampling utilities test completed!")