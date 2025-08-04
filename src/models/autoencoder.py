"""
Autoencoder model with clean imports and configuration.
"""
# Standard library
from dataclasses import dataclass
from typing import Optional

# Third-party imports
import tensorflow as tf
from omegaconf import DictConfig

# Local imports
from src.models.layers import Encoder, Decoder, DiagonalGaussian
from src.utils.logging import get_logger


@dataclass
class AutoEncoderConfig:
    """Type-safe configuration for AutoEncoder."""
    resolution: int = 256
    in_channels: int = 1
    ch: int = 2
    out_ch: int = 1
    ch_mult: list[int] = None
    num_res_blocks: int = 4
    z_channels: int = 4
    scale_factor: float = 1.0
    shift_factor: float = 0.0
    
    def __post_init__(self):
        if self.ch_mult is None:
            self.ch_mult = [2, 4, 8, 16]


class AutoEncoder(tf.keras.Model):
    """Variational AutoEncoder for neutron star data."""
    
    def __init__(self, config: AutoEncoderConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Build components
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
        
    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """Encode input to latent space."""
        z = self.regularizer(self.encoder(x))
        z = self.config.scale_factor * (z - self.config.shift_factor)
        return z

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """Decode from latent space."""
        z = z / self.config.scale_factor + self.config.shift_factor
        return self.decoder(z)

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass through autoencoder."""
        return self.decode(self.encode(x))
    
    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        return {
            "config": self.config.__dict__,
            **super().get_config()
        }
    
    @classmethod
    def from_config(cls, config_dict: dict):
        """Create model from configuration."""
        config = AutoEncoderConfig(**config_dict["config"])
        return cls(config)


def build_autoencoder(config: DictConfig) -> AutoEncoder:
    """Factory function to build autoencoder from Hydra config."""
    model_config = AutoEncoderConfig(
        resolution=config.model.resolution,
        in_channels=config.model.in_channels,
        ch=config.model.ch,
        out_ch=config.model.out_ch,
        ch_mult=config.model.ch_mult,
        num_res_blocks=config.model.num_res_blocks,
        z_channels=config.model.z_channels,
        scale_factor=config.model.scale_factor,
        shift_factor=config.model.shift_factor,
    )
    
    return AutoEncoder(model_config)