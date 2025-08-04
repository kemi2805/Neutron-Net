# src/models/autoencoder_pytorch.py
"""
PyTorch implementation of the AutoEncoder model for neutron star data.
Converted from TensorFlow/Keras implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List
import math


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation function."""
    return x * torch.sigmoid(x)


class Swish(nn.Module):
    """Swish activation as a module."""
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class GroupNorm(nn.Module):
    """Group Normalization layer."""
    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=min(num_groups, num_channels), 
            num_channels=num_channels, 
            eps=eps
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class AttnBlock(nn.Module):
    """Self-attention block for the autoencoder."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        self.norm = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.k = nn.Conv2d(channels, channels, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).transpose(1, 2)  # b, hw, c
        k = k.reshape(b, c, h * w)  # b, c, hw
        v = v.reshape(b, c, h * w).transpose(1, 2)  # b, hw, c
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(c)
        attn = torch.softmax(torch.bmm(q, k) * scale, dim=-1)
        h_ = torch.bmm(attn, v)  # b, hw, c
        h_ = h_.transpose(1, 2).reshape(b, c, h, w)
        
        return x + self.proj_out(h_)


class ResnetBlock(nn.Module):
    """ResNet block with group normalization and swish activation."""
    
    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        
        self.norm1 = GroupNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, 3, padding=1, bias=False)
        
        self.norm2 = GroupNorm(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=False)
        
        self.swish = Swish()
        
        if in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.out_channels, 1, bias=False)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.swish(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.swish(h)
        h = self.conv2(h)
        
        return self.shortcut(x) + h


class Downsample(nn.Module):
    """Downsampling layer with asymmetric padding."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Asymmetric padding: pad right and bottom by 1
        x = F.pad(x, (0, 1, 0, 1), mode='constant', value=0)
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer with nearest neighbor interpolation."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Encoder(nn.Module):
    """Encoder network for the VAE."""
    
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: List[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        self.num_resolutions = len(ch_mult)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)
        
        # Downsampling blocks
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            
            down = nn.ModuleDict({
                'block': block,
                'attn': nn.ModuleList()  # Empty for now, can add attention layers
            })
            
            if i_level != self.num_resolutions - 1:
                down['downsample'] = Downsample(block_in)
            
            self.down.append(down)
        
        # Middle blocks
        self.mid = nn.ModuleDict({
            'block_1': ResnetBlock(block_in, block_in),
            'attn_1': AttnBlock(block_in),
            'block_2': ResnetBlock(block_in, block_in),
        })
        
        # Output layers
        self.norm_out = GroupNorm(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial convolution
        h = self.conv_in(x)
        
        # Downsampling
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level]['block'][i_block](h)
            
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level]['downsample'](h)
        
        # Middle blocks
        h = self.mid['block_1'](h)
        h = self.mid['attn_1'](h)
        h = self.mid['block_2'](h)
        
        # Output
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        
        return h


class Decoder(nn.Module):
    """Decoder network for the VAE."""
    
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        out_ch: int,
        ch_mult: List[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.ch = ch
        self.out_ch = out_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        self.num_resolutions = len(ch_mult)
        
        # Initial convolution
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, padding=1)
        
        # Middle blocks
        self.mid = nn.ModuleDict({
            'block_1': ResnetBlock(block_in, block_in),
            'attn_1': AttnBlock(block_in),
            'block_2': ResnetBlock(block_in, block_in),
        })
        
        # Upsampling blocks
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            
            up = nn.ModuleDict({
                'block': block,
                'attn': nn.ModuleList()
            })
            
            if i_level != 0:
                up['upsample'] = Upsample(block_in)
            
            self.up.append(up)
        
        # Output layers
        self.norm_out = GroupNorm(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Initial convolution
        h = self.conv_in(z)
        
        # Middle blocks
        h = self.mid['block_1'](h)
        h = self.mid['attn_1'](h)
        h = self.mid['block_2'](h)
        
        # Upsampling
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level]['block'][i_block](h)
            
            if i_level != self.num_resolutions - 1:
                h = self.up[i_level]['upsample'](h)
        
        # Output
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        
        return h


class DiagonalGaussian(nn.Module):
    """Diagonal Gaussian distribution for VAE latent space."""
    
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into mean and logvar
        mean, logvar = torch.chunk(x, 2, dim=1)
        
        if self.sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            return mean


@dataclass
class AutoEncoderConfig:
    """Configuration for the AutoEncoder."""
    resolution: int = 256
    in_channels: int = 1
    ch: int = 2
    out_ch: int = 1
    ch_mult: List[int] = None
    num_res_blocks: int = 4
    z_channels: int = 4
    scale_factor: float = 1.0
    shift_factor: float = 0.0
    
    def __post_init__(self):
        if self.ch_mult is None:
            self.ch_mult = [2, 4, 8, 16]


class AutoEncoder(nn.Module):
    """Complete AutoEncoder model."""
    
    def __init__(self, config: AutoEncoderConfig):
        super().__init__()
        self.config = config
        
        self.encoder = Encoder(
            resolution=config.resolution,
            in_channels=config.in_channels,
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            z_channels=config.z_channels,
        )
        
        self.decoder = Decoder(
            resolution=config.resolution,
            in_channels=config.in_channels,
            ch=config.ch,
            out_ch=config.out_ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            z_channels=config.z_channels,
        )
        
        self.regularizer = DiagonalGaussian()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        z = self.regularizer(self.encoder(x))
        z = self.config.scale_factor * (z - self.config.shift_factor)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        z = z / self.config.scale_factor + self.config.shift_factor
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        return self.decode(self.encode(x))


# Example usage
if __name__ == "__main__":
    # Create model configuration
    config = AutoEncoderConfig(
        resolution=256,
        in_channels=1,
        ch=2,
        out_ch=1,
        ch_mult=[2, 4, 8, 16],
        num_res_blocks=4,
        z_channels=4,
    )
    
    # Create model
    model = AutoEncoder(config)
    
    # Test with dummy input
    x = torch.randn(2, 1, 256, 256)  # batch_size=2, channels=1, H=256, W=256
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")