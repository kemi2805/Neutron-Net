#!/usr/bin/env python3
"""
Main training script for neutron star diffusion model in PyTorch.
Converted from TensorFlow implementation.

Usage:
    python main_pytorch.py --config config.yaml
    python main_pytorch.py --data_path /path/to/data.npy --epochs 100
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.autoencoder import AutoEncoderConfig
from src.models.diffusion import NeutronNet, CosineBetaScheduler
from src.data.data_loader import load_neutron_star_data
from src.training.trainer import NeutronStarTrainer, TrainingConfig
from src.utils.physics import validate_physics_constraints, plot_physics_validation_results


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device() -> torch.device:
    """Setup computing device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def create_model(config: Dict[str, Any], device: torch.device) -> NeutronNet:
    """Create and initialize the diffusion model."""
    
    # Create autoencoder config
    ae_config = AutoEncoderConfig(
        resolution=config.get('resolution', 256),
        in_channels=config.get('in_channels', 1),
        ch=config.get('ch', 2),
        out_ch=config.get('out_ch', 1),
        ch_mult=config.get('ch_mult', [2, 4, 8, 16]),
        num_res_blocks=config.get('num_res_blocks', 4),
        z_channels=config.get('z_channels', 4),
        scale_factor=config.get('scale_factor', 1.0),
        shift_factor=config.get('shift_factor', 0.0)
    )
    
    # Create beta schedule
    scheduler = CosineBetaScheduler(
        num_timesteps=config.get('num_timesteps', 1000),
        s=config.get('schedule_s', 0.008)
    )
    beta_schedule = scheduler.get_schedule()
    
    # Create model
    model = NeutronNet(
        ae_config,
        beta_schedule,
        accumulation_steps=config.get('accumulation_steps', 4)
    )
    
    model.to(device)
    
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train neutron star diffusion model")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_path', type=str, help='Path to training data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--validate_only', action='store_true', help='Only run validation')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            'resolution': 256,
            'in_channels': 1,
            'ch': 2,
            'out_ch': 1,
            'ch_mult': [2, 4, 8, 16],
            'num_res_blocks': 4,
            'z_channels': 4,
            'scale_factor': 1.0,
            'shift_factor': 0.0,
            'num_timesteps': 1000,
            'schedule_s': 0.008,
            'accumulation_steps': 4,
            'data_path': '/mnt/rafast/miler/ml_data_pics.npy',
            'validation_split': 0.1,
            'num_workers': 4
        }
    
    # Override config with command line arguments
    if args.data_path:
        config['data_path'] = args.data_path
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Setup device
    device = setup_device()
    
    # Load data
    print("\nLoading data...")
    try:
        train_loader, val_loader = load_neutron_star_data(
            data_path=config['data_path'],
            validation_split=config.get('validation_split', 0.1),
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4),
            augment=True
        )
        print(f"Data loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, device)
    
    # Training configuration
    training_config = TrainingConfig(
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        accumulation_steps=config.get('accumulation_steps', 4),
        weight_decay=config.get('weight_decay', 1e-4),
        use_scheduler=config.get('use_scheduler', True),
        scheduler_patience=config.get('scheduler_patience', 10),
        scheduler_factor=config.get('scheduler_factor', 0.5),
        validation_interval=config.get('validation_interval', 5),
        save_interval=config.get('save_interval', 10),
        log_interval=config.get('log_interval', 100),
        use_wandb=args.use_wandb,
        use_tensorboard=config.get('use_tensorboard', True),
        checkpoint_dir=config['checkpoint_dir'],
        log_dir=config.get('log_dir', 'logs')
    )
    
    # Create trainer
    trainer = NeutronStarTrainer(model, training_config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Validation only mode
    if args.validate_only:
        print("\nRunning validation only...")
        val_metrics = trainer.validate(val_loader)
        
        print("Validation metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Physics validation
        print("\nRunning physics validation...")
        sample_batch = next(iter(val_loader))
        physics_metrics = validate_physics_constraints(
            model.model,  # Use autoencoder part
            sample_batch.numpy()
        )
        
        print("Physics validation metrics:")
        for key, value in physics_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Plot results
        plot_physics_validation_results(
            physics_metrics,
            Path(training_config.log_dir) / "physics_validation.png"
        )
        
        return
    
    # Training
    print(f"\nStarting training...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Training for {training_config.epochs} epochs")
    
    try:
        trainer.train(train_loader, val_loader)
        
        # Plot training history
        history_path = Path(training_config.log_dir) / "training_history.png"
        trainer.plot_training_history(str(history_path))
        
        print(f"\nTraining completed successfully!")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Checkpoints saved in: {training_config.checkpoint_dir}")
        print(f"Logs saved in: {training_config.log_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current state
        trainer.save_checkpoint()
        print("Checkpoint saved")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


def test_model():
    """Test the model with dummy data."""
    print("Testing model with dummy data...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy config
    config = {
        'resolution': 64,  # Smaller for testing
        'in_channels': 1,
        'ch': 2,
        'out_ch': 1,
        'ch_mult': [2, 4],  # Simpler architecture
        'num_res_blocks': 2,
        'z_channels': 4,
        'accumulation_steps': 2
    }
    
    # Create model
    model = create_model(config, device)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 64, 64, device=device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        # Test autoencoder
        reconstructed = model.model(dummy_input)
        print(f"Autoencoder output shape: {reconstructed.shape}")
        
        # Test diffusion forward pass
        t = torch.randint(0, len(model.beta_schedule), (batch_size,), device=device)
        noisy_images, noise = model.forward_diffusion_step(dummy_input, t)
        print(f"Noisy images shape: {noisy_images.shape}")
        
        predicted_noise = model.reverse_diffusion_step(noisy_images, t)
        print(f"Predicted noise shape: {predicted_noise.shape}")
    
    print("Model test completed successfully!")


if __name__ == "__main__":
    # Add test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_model()
    else:
        main()