"""
Professional logging setup for neutron star diffusion project.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

import wandb
from omegaconf import DictConfig
from rich.console import Console
from rich.logging import RichHandler


class ProjectLogger:
    """Centralized logging for the entire project."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.console = Console()
        self._setup_logging()
        self._setup_wandb() if config.logging.use_wandb else None
    
    def _setup_logging(self):
        """Setup rich logging with file output."""
        # Create logs directory
        log_dir = Path(self.config.data.logs_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                RichHandler(console=self.console, rich_tracebacks=True),
                logging.FileHandler(log_dir / f"{self.config.experiment_name}.log")
            ]
        )
        
        # Create project logger
        self.logger = logging.getLogger("neutron_star_diffusion")
        
    def _setup_wandb(self):
        """Initialize Weights & Biases."""
        try:
            wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=dict(self.config),
                dir=self.config.data.logs_dir
            )
            self.logger.info("✅ Weights & Biases initialized")
        except Exception as e:
            self.logger.warning(f"❌ Failed to initialize W&B: {e}")
    
    def log_model_summary(self, model, model_name: str):
        """Log model architecture."""
        self.logger.info(f"📋 {model_name} Architecture:")
        
        # Count parameters
        total_params = sum(p.numpy().size for p in model.trainable_variables)
        self.logger.info(f"   Total Parameters: {total_params:,}")
        
        # Log to wandb if available
        if wandb.run:
            wandb.log({f"{model_name}_total_params": total_params})
    
    def log_training_start(self, train_size: int, val_size: int):
        """Log training initialization."""
        self.logger.info("🚀 Starting Training")
        self.logger.info(f"   Training samples: {train_size:,}")
        self.logger.info(f"   Validation samples: {val_size:,}")
        self.logger.info(f"   Batch size: {self.config.training.batch_size}")
        self.logger.info(f"   Epochs: {self.config.training.epochs}")
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None):
        """Log epoch results."""
        msg = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}"
        if val_loss:
            msg += f" | Val Loss: {val_loss:.4f}"
        
        self.logger.info(msg)
        
        # Log to wandb
        if wandb.run:
            log_dict = {"epoch": epoch, "train_loss": train_loss}
            if val_loss:
                log_dict["val_loss"] = val_loss
            wandb.log(log_dict)
    
    def log_physics_validation(self, metrics: dict):
        """Log physics-specific validation metrics."""
        self.logger.info("🔬 Physics Validation:")
        for metric_name, value in metrics.items():
            self.logger.info(f"   {metric_name}: {value:.4f}")
        
        if wandb.run:
            wandb.log({f"physics_{k}": v for k, v in metrics.items()})
    
    def log_error(self, error: Exception, context: str):
        """Log errors with context."""
        self.logger.error(f"❌ Error in {context}: {str(error)}", exc_info=True)
    
    def log_checkpoint_saved(self, path: str, epoch: int):
        """Log checkpoint saving."""
        self.logger.info(f"💾 Checkpoint saved: {path} (epoch {epoch})")
    
    def close(self):
        """Clean up logging."""
        if wandb.run:
            wandb.finish()
        self.logger.info("🏁 Training completed")


def get_logger(config: DictConfig) -> ProjectLogger:
    """Factory function to get project logger."""
    return ProjectLogger(config)