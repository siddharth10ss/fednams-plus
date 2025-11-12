"""Complete FedNAMs model combining feature extraction and NAM head."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .feature_extractor import FeatureExtractor
from .nam_head import NAMHead
from configs.config import ModelConfig
from utils.exceptions import ModelError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class FedNAMsModel(nn.Module):
    """Complete FedNAMs model for interpretable federated learning.
    
    Combines CNN feature extraction with Neural Additive Model head
    for interpretable predictions on medical images.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize FedNAMs model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        
        # Feature extractor (CNN backbone)
        self.feature_extractor = FeatureExtractor(
            backbone=config.backbone,
            pretrained=config.pretrained,
            feature_dim=config.feature_dim
        )
        
        # NAM head
        self.nam_head = NAMHead(
            feature_dim=config.feature_dim,
            num_classes=config.num_classes,
            hidden_units=config.nam_hidden_units,
            dropout=config.dropout,
            use_exu=config.use_exu
        )
        
        logger.info(
            f"FedNAMsModel initialized: {config.backbone}, "
            f"{config.feature_dim}D features, {config.num_classes} classes"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass.
        
        Args:
            x: Input images (batch_size, 3, height, width)
            
        Returns:
            Predictions (batch_size, num_classes)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Get predictions
        predictions, _ = self.nam_head(features)
        
        return predictions
    
    def forward_with_contributions(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass with feature contributions for interpretability.
        
        Args:
            x: Input images (batch_size, 3, height, width)
            
        Returns:
            Tuple of:
                - predictions: Final predictions (batch_size, num_classes)
                - features: Extracted features (batch_size, feature_dim)
                - contributions: List of feature contributions
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Get predictions and contributions
        predictions, contributions = self.nam_head(features)
        
        return predictions, features, contributions
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for federated communication.
        
        Returns:
            Dictionary of parameter names to tensors
        """
        return {name: param.data.clone() for name, param in self.named_parameters()}
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set model parameters from federated aggregation.
        
        Args:
            params: Dictionary of parameter names to tensors
        """
        for name, param in self.named_parameters():
            if name in params:
                param.data = params[name].clone()
            else:
                logger.warning(f"Parameter {name} not found in provided params")
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_feature_extractor(self):
        """Freeze feature extractor for faster training."""
        self.feature_extractor.freeze_backbone()
    
    def unfreeze_feature_extractor(self):
        """Unfreeze feature extractor for fine-tuning."""
        self.feature_extractor.unfreeze_backbone()
    
    def get_model_summary(self) -> Dict[str, any]:
        """Get model summary information.
        
        Returns:
            Dictionary with model information
        """
        return {
            'backbone': self.config.backbone,
            'feature_dim': self.config.feature_dim,
            'num_classes': self.config.num_classes,
            'nam_hidden_units': self.config.nam_hidden_units,
            'total_parameters': self.get_num_parameters(),
            'trainable_parameters': self.get_num_trainable_parameters(),
            'feature_extractor_params': self.feature_extractor.get_num_parameters(),
            'nam_head_params': self.nam_head.get_num_parameters()
        }
