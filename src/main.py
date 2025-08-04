"""
Main entry point for neutron star diffusion training.
"""
import os
from pathlib import Path

import hydra
import tensorflow as tf
from omegaconf import DictConfig

from src.data.loaders import load_neutron_star_data
from src.models.autoencoder import build_autoencoder
from src.training.trainer import AutoEncoderTrainer
from src.utils.logging import get_logger
from src.utils.physics import validate_physics_constraints


# Set TensorFlow to be deterministic
os.environ['TF_DETERMINISTIC_OPS'] = '1'


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Setup
    tf.random.set_seed(cfg.random_seed)
    if cfg.hardware.gpu_memory_growth:
        _setup_gpu()
    
    # Initialize logging
    logger = get_logger(cfg)
    logger.logger.info("🌟 Starting Neutron Star Diffusion Training")
    
    try:
        # Load data
        logger.logger.info("📊 Loading data...")
        train_data, val_data = load_neutron_star_data(cfg)
        logger.log_training_start(len(train_data), len(val_data))
        
        # Build model
        logger.logger.info("🏗️ Building model...")
        model = build_autoencoder(cfg)
        logger.log_model_summary(model, "AutoEncoder")
        
        # Initialize trainer
        trainer = AutoEncoderTrainer(cfg, logger)
        
        # Train model
        logger.logger.info("🚀 Starting training...")
        history = trainer.fit(model, train_data, val_data)
        
        # Physics validation
        logger.logger.info("🔬 Running physics validation...")
        physics_metrics = validate_physics_constraints(model, val_data[:100])
        logger.log_physics_validation(physics_metrics)
        
        logger.logger.info("✅ Training completed successfully!")
        
    except Exception as e:
        logger.log_error(e, "main training loop")
        raise
    finally:
        logger.close()


def _setup_gpu():
    """Configure GPU memory growth."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU setup error: {e}")


if __name__ == "__main__":
    main()