# src/training/trainer.py
"""
PyTorch trainer for neutron star diffusion model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm.auto import tqdm
import os
from pathlib import Path
from typing import Dict, Optional, List, Any
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from ..models.diffusion import NeutronNet, DiffusionTrainer
from ..models.autoencoder import AutoEncoderConfig
from ..utils.physics import validate_physics_constraints


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 8
    accumulation_steps: int = 4
    weight_decay: float = 1e-4
    
    # Scheduling
    use_scheduler: bool = True
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Validation
    validation_interval: int = 5
    save_interval: int = 10
    
    # Logging
    log_interval: int = 100
    use_wandb: bool = False
    use_tensorboard: bool = True
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


class NeutronStarTrainer:
    """Main trainer class for neutron star diffusion model."""
    
    def __init__(
        self,
        model: NeutronNet,
        config: TrainingConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        if config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=config.scheduler_patience,
                factor=config.scheduler_factor,
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Setup loss function
        from ..models.diffusion import PhysicsConstraintLoss
        self.loss_fn = PhysicsConstraintLoss()
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def setup_logging(self):
        """Setup logging with tensorboard and wandb."""
        # Create directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Tensorboard
        if self.config.use_tensorboard:
            self.tb_writer = SummaryWriter(self.config.log_dir)
        else:
            self.tb_writer = None
        
        # Weights & Biases
        if self.config.use_wandb:
            wandb.init(
                project="neutron-star-diffusion",
                config=self.config.__dict__
            )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            
            # Training step
            loss = self.train_step(batch)
            total_loss += loss
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                    'lr': f"{current_lr:.2e}"
                })
                
                # Log to tensorboard
                if self.tb_writer:
                    step = self.current_epoch * num_batches + batch_idx
                    self.tb_writer.add_scalar('train/batch_loss', loss, step)
                    self.tb_writer.add_scalar('train/learning_rate', current_lr, step)
                
                # Log to wandb
                if self.config.use_wandb:
                    wandb.log({
                        'train/batch_loss': loss,
                        'train/learning_rate': current_lr,
                        'epoch': self.current_epoch
                    })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step with gradient accumulation."""
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        batch_size = batch.shape[0]
        sub_batch_size = max(1, batch_size // self.config.accumulation_steps)
        
        for i in range(self.config.accumulation_steps):
            start_idx = i * sub_batch_size
            end_idx = min((i + 1) * sub_batch_size, batch_size)
            
            if start_idx >= batch_size:
                break
                
            sub_batch = batch[start_idx:end_idx]
            
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
            loss = self.loss_fn(predicted_noise, sub_batch) / self.config.accumulation_steps
            
            # Backward pass
            loss.backward()
            total_loss += loss.item() * self.config.accumulation_steps
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        
        return total_loss
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        for batch in tqdm(val_loader, desc="Validating"):
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
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        # Physics validation on a subset
        physics_metrics = validate_physics_constraints(
            self.model.model,  # Use the autoencoder part
            batch[:min(10, batch.shape[0])].cpu().numpy()
        )
        
        return {
            'val_loss': avg_loss,
            **physics_metrics
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """Main training loop."""
        print(f"Starting training for {self.config.epochs} epochs")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Validation
            if val_loader and epoch % self.config.validation_interval == 0:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['val_loss']
                self.training_history['val_loss'].append(val_loss)
                
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
                
                # Log validation metrics
                if self.tb_writer:
                    for key, value in val_metrics.items():
                        self.tb_writer.add_scalar(f'val/{key}', value, epoch)
                
                if self.config.use_wandb:
                    wandb.log({f'val/{k}': v for k, v in val_metrics.items()})
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
            
            # Log training metrics
            if self.tb_writer:
                self.tb_writer.add_scalar('train/epoch_loss', train_loss, epoch)
            
            if self.config.use_wandb:
                wandb.log({'train/epoch_loss': train_loss, 'epoch': epoch})
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint()
        
        print("Training completed!")
        
        # Close logging
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.config.use_wandb:
            wandb.finish()
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.training_history['train_loss'], label='Train Loss')
        if self.training_history['val_loss']:
            val_epochs = list(range(0, len(self.training_history['train_loss']), 
                                  self.config.validation_interval))
            axes[0].plot(val_epochs, self.training_history['val_loss'], 
                        label='Val Loss', marker='o')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Learning rate plot
        axes[1].plot(self.training_history['learning_rate'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage
if __name__ == "__main__":
    from ..models.diffusion import CosineBetaScheduler
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader = load_neutron_star_data(
        "/path/to/ml_data_pics.npy",
        batch_size=8,
        validation_split=0.1
    )
    
    # Create model
    ae_config = AutoEncoderConfig(
        resolution=256,
        in_channels=1,
        ch=2,
        out_ch=1,
        ch_mult=[2, 4, 8, 16],
        num_res_blocks=4,
        z_channels=4,
    )
    
    scheduler = CosineBetaScheduler(1000)
    beta_schedule = scheduler.get_schedule()
    
    model = NeutronNet(ae_config, beta_schedule, accumulation_steps=4)
    model.to(device)
    
    # Training config
    training_config = TrainingConfig(
        epochs=100,
        learning_rate=1e-3,
        batch_size=8,
        use_wandb=False,
        use_tensorboard=True
    )
    
    # Create trainer and train
    trainer = NeutronStarTrainer(model, training_config, device)
    trainer.train(train_loader, val_loader)
    
    # Plot results
    trainer.plot_training_history("training_history.png")