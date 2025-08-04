# src/data/augmentation_pytorch.py
"""
PyTorch augmentation module for neutron star diffusion.
Converted from TensorFlow/Keras implementation.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import os
from pathlib import Path
from typing import Optional, Tuple, Union, List
import cv2


class NeutronStarAugmentation:
    """
    Comprehensive augmentation pipeline for neutron star data.
    Designed to preserve physical properties while adding variation.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        crop_size: Tuple[int, int] = (200, 200),
        rotation_degrees: float = 20.0,
        translation_factor: float = 0.2,
        zoom_factor: Tuple[float, float] = (0.7, 1.2),
        contrast_factor: float = 0.5,
        brightness_factor: float = 0.3,
        noise_std: float = 0.02,
        preserve_physics: bool = True
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            image_size: Target image size (H, W)
            crop_size: Random crop size (H, W)
            rotation_degrees: Maximum rotation in degrees
            translation_factor: Translation as fraction of image size
            zoom_factor: Zoom range (min, max)
            contrast_factor: Contrast adjustment factor
            brightness_factor: Brightness adjustment factor
            noise_std: Standard deviation for Gaussian noise
            preserve_physics: Whether to apply physics-preserving constraints
        """
        self.image_size = image_size
        self.crop_size = crop_size
        self.rotation_degrees = rotation_degrees
        self.translation_factor = translation_factor
        self.zoom_factor = zoom_factor
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor
        self.noise_std = noise_std
        self.preserve_physics = preserve_physics
        
        # Define individual augmentation transforms
        self.transforms = self._create_transforms()
    
    def _create_transforms(self) -> List:
        """Create list of individual augmentation transforms."""
        transforms_list = []
        
        # Random crop
        if self.crop_size != self.image_size:
            transforms_list.append(RandomCrop(self.crop_size))
        
        # Geometric transforms
        transforms_list.extend([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(self.rotation_degrees),
            RandomTranslation(self.translation_factor),
            RandomZoom(self.zoom_factor),
        ])
        
        # Photometric transforms
        transforms_list.extend([
            RandomContrast(self.contrast_factor),
            RandomBrightness(self.brightness_factor),
            GaussianNoise(std=self.noise_std),
        ])
        
        return transforms_list
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation to image."""
        return self.apply_random_augmentation(image)
    
    def apply_random_augmentation(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply a randomly selected augmentation to the image.
        
        Args:
            image: Input tensor of shape (C, H, W) or (H, W)
            
        Returns:
            Augmented image tensor
        """
        # Ensure image has channel dimension
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Add channel dimension
        
        # Randomly select and apply augmentation
        augmentation = random.choice(self.transforms)
        augmented = augmentation(image)
        
        # Apply physics constraints if enabled
        if self.preserve_physics:
            augmented = self._apply_physics_constraints(augmented)
        
        return augmented
    
    def apply_multiple_augmentations(
        self, 
        image: torch.Tensor, 
        num_augmentations: int = 2
    ) -> torch.Tensor:
        """Apply multiple random augmentations."""
        augmented = image.clone()
        
        for _ in range(num_augmentations):
            # Apply with some probability to avoid over-augmentation
            if random.random() < 0.7:
                augmentation = random.choice(self.transforms)
                augmented = augmentation(augmented)
        
        if self.preserve_physics:
            augmented = self._apply_physics_constraints(augmented)
        
        return augmented
    
    def _apply_physics_constraints(self, image: torch.Tensor) -> torch.Tensor:
        """Apply physics-based constraints to preserve neutron star properties."""
        # Ensure values remain in [0, 1] range
        image = torch.clamp(image, 0, 1)
        
        # Preserve energy conservation (optional)
        original_sum = image.sum()
        if original_sum > 0:
            # Maintain similar total energy
            current_sum = image.sum()
            if current_sum > 0:
                scaling_factor = min(1.2, max(0.8, original_sum / current_sum))
                image = image * scaling_factor
                image = torch.clamp(image, 0, 1)
        
        return image


# Individual augmentation classes
class RandomCrop:
    """Random crop augmentation."""
    
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random crop."""
        _, h, w = image.shape
        target_h, target_w = self.size
        
        if h <= target_h and w <= target_w:
            return image
        
        # Random crop coordinates
        top = random.randint(0, max(0, h - target_h))
        left = random.randint(0, max(0, w - target_w))
        
        cropped = image[:, top:top + target_h, left:left + target_w]
        
        # Resize to original size if needed
        if cropped.shape[1:] != (h, w):
            cropped = F.interpolate(
                cropped.unsqueeze(0), 
                size=(h, w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        return cropped


class RandomHorizontalFlip:
    """Random horizontal flip."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return torch.flip(image, dims=[-1])
        return image


class RandomVerticalFlip:
    """Random vertical flip."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return torch.flip(image, dims=[-2])
        return image


class RandomRotation:
    """Random rotation augmentation."""
    
    def __init__(self, degrees: float):
        self.degrees = degrees
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(-self.degrees, self.degrees)
        
        # Convert to PIL for rotation, then back to tensor
        if image.shape[0] == 1:  # Single channel
            pil_image = transforms.ToPILImage()(image)
            rotated = transforms.functional.rotate(pil_image, angle, fill=0)
            return transforms.ToTensor()(rotated)
        else:
            # Handle multi-channel
            rotated_channels = []
            for c in range(image.shape[0]):
                pil_image = transforms.ToPILImage()(image[c:c+1])
                rotated = transforms.functional.rotate(pil_image, angle, fill=0)
                rotated_channels.append(transforms.ToTensor()(rotated))
            return torch.cat(rotated_channels, dim=0)


class RandomTranslation:
    """Random translation augmentation."""
    
    def __init__(self, factor: float):
        self.factor = factor
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        _, h, w = image.shape
        
        # Random translation
        max_dx = int(self.factor * w)
        max_dy = int(self.factor * h)
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        
        # Create affine transformation matrix
        theta = torch.tensor([
            [1, 0, 2 * dx / w],
            [0, 1, 2 * dy / h]
        ], dtype=torch.float32).unsqueeze(0)
        
        # Apply transformation
        grid = F.affine_grid(theta, image.unsqueeze(0).shape, align_corners=False)
        translated = F.grid_sample(
            image.unsqueeze(0), 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        )
        
        return translated.squeeze(0)


class RandomZoom:
    """Random zoom augmentation."""
    
    def __init__(self, zoom_range: Tuple[float, float]):
        self.zoom_range = zoom_range
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        zoom_factor = random.uniform(*self.zoom_range)
        
        if zoom_factor == 1.0:
            return image
        
        # Calculate new size
        _, h, w = image.shape
        new_h = int(h * zoom_factor)
        new_w = int(w * zoom_factor)
        
        # Resize image
        zoomed = F.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Crop or pad to original size
        if zoom_factor > 1.0:  # Zoom in - crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            zoomed = zoomed[:, start_h:start_h + h, start_w:start_w + w]
        else:  # Zoom out - pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            zoomed = F.pad(zoomed, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
            
            # Ensure exact size
            if zoomed.shape[1] != h or zoomed.shape[2] != w:
                zoomed = F.interpolate(
                    zoomed.unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
        
        return zoomed


class RandomContrast:
    """Random contrast adjustment."""
    
    def __init__(self, factor: float):
        self.factor = factor
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # Random contrast factor
        contrast = random.uniform(1 - self.factor, 1 + self.factor)
        
        # Apply contrast adjustment
        mean = image.mean()
        contrasted = (image - mean) * contrast + mean
        
        return torch.clamp(contrasted, 0, 1)


class RandomBrightness:
    """Random brightness adjustment."""
    
    def __init__(self, factor: float):
        self.factor = factor
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # Random brightness adjustment
        brightness = random.uniform(-self.factor, self.factor)
        
        brightened = image + brightness
        return torch.clamp(brightened, 0, 1)


class GaussianNoise:
    """Add Gaussian noise."""
    
    def __init__(self, std: float = 0.02):
        self.std = std
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(image) * self.std
        noisy = image + noise
        return torch.clamp(noisy, 0, 1)


# Utility functions
def create_test_image(height: int = 256, width: int = 256) -> torch.Tensor:
    """Create a test image with gradient pattern."""
    x = torch.linspace(0, 1, width)
    y = torch.linspace(0, 1, height)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    # Create gradient pattern
    gradient = (X + Y) / 2
    
    # Add some structure to mimic neutron star data
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 4
    
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height), 
        torch.arange(width), 
        indexing='ij'
    )
    
    # Create circular pattern
    dist = torch.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    mask = dist < radius
    
    gradient[mask] = gradient[mask] * 1.5
    gradient = torch.clamp(gradient, 0, 1)
    
    return gradient.unsqueeze(0)  # Add channel dimension


def augment_and_save_batch(
    images: torch.Tensor,
    output_folder: str,
    num_augments_per_image: int = 5,
    augmentation_pipeline: Optional[NeutronStarAugmentation] = None
):
    """
    Apply augmentations to a batch of images and save results.
    
    Args:
        images: Tensor of shape (N, C, H, W)
        output_folder: Directory to save augmented images
        num_augments_per_image: Number of augmentations per input image
        augmentation_pipeline: Custom augmentation pipeline
    """
    if augmentation_pipeline is None:
        augmentation_pipeline = NeutronStarAugmentation()
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for img_idx, image in enumerate(images):
        # Save original
        original_path = output_path / f"original_{img_idx:03d}.png"
        save_tensor_as_image(image, original_path)
        
        # Generate and save augmentations
        for aug_idx in range(num_augments_per_image):
            augmented = augmentation_pipeline(image)
            aug_path = output_path / f"augmented_{img_idx:03d}_{aug_idx:02d}.png"
            save_tensor_as_image(augmented, aug_path)
    
    print(f"Saved {len(images)} original and {len(images) * num_augments_per_image} augmented images to {output_folder}")


def save_tensor_as_image(tensor: torch.Tensor, path: Union[str, Path]):
    """Save tensor as image file."""
    # Convert to numpy
    if len(tensor.shape) == 3 and tensor.shape[0] == 1:
        # Single channel
        image_np = tensor.squeeze(0).cpu().numpy()
    elif len(tensor.shape) == 2:
        image_np = tensor.cpu().numpy()
    else:
        # Multi-channel - take first channel
        image_np = tensor[0].cpu().numpy()
    
    # Convert to 0-255 range
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
    
    # Save using PIL
    Image.fromarray(image_np, mode='L').save(path)


# Compose multiple transforms
class Compose:
    """Compose multiple augmentation transforms."""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            image = transform(image)
        return image


# Example usage and testing
if __name__ == "__main__":
    # Create test image
    test_image = create_test_image(256, 256)
    print(f"Test image shape: {test_image.shape}")
    
    # Create augmentation pipeline
    aug_pipeline = NeutronStarAugmentation(
        image_size=(256, 256),
        preserve_physics=True
    )
    
    # Test single augmentation
    augmented = aug_pipeline(test_image)
    print(f"Augmented image shape: {augmented.shape}")
    
    # Test batch augmentation
    batch = test_image.unsqueeze(0).repeat(4, 1, 1, 1)  # Create batch of 4
    print(f"Batch shape: {batch.shape}")
    
    # Save examples
    augment_and_save_batch(
        batch,
        "test_augmentations",
        num_augments_per_image=3,
        augmentation_pipeline=aug_pipeline
    )
    
    print("Augmentation test completed!")