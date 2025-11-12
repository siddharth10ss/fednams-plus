"""Local trainer for client-side training in federated learning."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from pathlib import Path

from configs.config import TrainingConfig
from utils.exceptions import TrainingError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class LocalTrainer:
    """Manages local training on each federated client.
    
    Handles training loops, optimization, and metric tracking for a single client.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = 'cuda'
    ):
        """Initialize local trainer.
        
        Args:
            model: Neural network model to train
            config: Training configuration
            device: Device for training ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss function (multi-label classification)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Mixed precision training
        self.use_amp = config.mixed_precision and device == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(
            f"LocalTrainer initialized: {config.optimizer}, "
            f"lr={config.learning_rate}, amp={self.use_amp}"
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config.
        
        Returns:
            Optimizer instance
        """
        if self.config.optimizer == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
        elif self.config.optimizer == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9
            )
        elif self.config.optimizer == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
        else:
            raise TrainingError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler.
        
        Returns:
            Scheduler instance or None
        """
        if self.config.scheduler is None:
            return None
        
        if self.config.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_local_epochs
            )
        elif self.config.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            logger.warning(f"Unknown scheduler: {self.config.scheduler}")
            return None
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Compute average metrics
        avg_loss = total_loss / total_samples
        
        self.current_epoch += 1
        
        return {
            'loss': avg_loss,
            'epoch': self.current_epoch,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation/test data.
        
        Args:
            dataloader: Evaluation data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                
                # Store predictions and labels
                predictions = torch.sigmoid(outputs)
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
        
        # Compute metrics
        avg_loss = total_loss / total_samples
        
        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute accuracy (threshold at 0.5)
        pred_binary = (all_predictions > 0.5).float()
        accuracy = (pred_binary == all_labels).float().mean().item()
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """Check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if should stop, False otherwise
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {self.patience_counter} epochs")
                return True
            return False
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for federated aggregation.
        
        Returns:
            Dictionary of parameter names to tensors
        """
        return {
            name: param.data.clone().cpu()
            for name, param in self.model.named_parameters()
        }
    
    def set_model_parameters(self, params: Dict[str, torch.Tensor]):
        """Set model parameters from federated aggregation.
        
        Args:
            params: Dictionary of parameter names to tensors
        """
        for name, param in self.model.named_parameters():
            if name in params:
                param.data = params[name].to(self.device).clone()
            else:
                logger.warning(f"Parameter {name} not found in provided params")
    
    def save_checkpoint(self, path: Path):
        """Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.patience_counter = checkpoint['patience_counter']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path}")
