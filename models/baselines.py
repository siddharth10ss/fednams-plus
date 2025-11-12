"""Baseline models for comparison with FedNAMs+."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
import numpy as np

from configs.config import ModelConfig
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class FedAvgCNN(nn.Module):
    """Standard CNN baseline for federated learning.
    
    Uses standard CNN architecture without interpretability features.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize FedAvg CNN.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        
        # Load backbone
        if config.backbone == 'resnet18':
            self.model = models.resnet18(pretrained=config.pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, config.num_classes)
            
        elif config.backbone == 'resnet50':
            self.model = models.resnet50(pretrained=config.pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, config.num_classes)
            
        elif config.backbone == 'densenet121':
            self.model = models.densenet121(pretrained=config.pretrained)
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features,
                config.num_classes
            )
        
        logger.info(f"FedAvgCNN initialized: {config.backbone}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input images (batch_size, 3, height, width)
            
        Returns:
            Predictions (batch_size, num_classes)
        """
        return self.model(x)
    
    def get_parameters(self) -> dict:
        """Get model parameters."""
        return {name: param.data.clone() for name, param in self.named_parameters()}
    
    def set_parameters(self, params: dict):
        """Set model parameters."""
        for name, param in self.named_parameters():
            if name in params:
                param.data = params[name].clone()


class FedAvgCNN_GradCAM(FedAvgCNN):
    """FedAvg CNN with Grad-CAM explanation capability.
    
    Extends FedAvgCNN to support Grad-CAM visualizations.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize FedAvg CNN with Grad-CAM.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Store gradients and activations for Grad-CAM
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
        logger.info("FedAvgCNN_GradCAM initialized with Grad-CAM support")
    
    def _register_hooks(self):
        """Register forward and backward hooks for Grad-CAM."""
        if self.config.backbone.startswith('resnet'):
            target_layer = self.model.layer4[-1]
        elif self.config.backbone.startswith('densenet'):
            target_layer = self.model.features.denseblock4
        else:
            return
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def get_explanations(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM explanations.
        
        Args:
            x: Input images (batch_size, 3, height, width)
            class_idx: Target class index (None for predicted class)
            
        Returns:
            Grad-CAM heatmaps (batch_size, height, width)
        """
        self.eval()
        
        # Forward pass
        output = self.forward(x)
        
        # Get target class
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, class_idx.unsqueeze(1), 1.0)
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute Grad-CAM
        gradients = self.gradients.data
        activations = self.activations.data
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = nn.functional.interpolate(
            cam,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return cam.squeeze(1).cpu().numpy()


class CentralizedNAM(nn.Module):
    """Centralized NAM baseline (no federated learning).
    
    Same architecture as FedNAMs but trained on centralized data.
    Used as privacy baseline.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize Centralized NAM.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        # Import here to avoid circular dependency
        from .feature_extractor import FeatureExtractor
        from .nam_head import NAMHead
        
        self.config = config
        
        self.feature_extractor = FeatureExtractor(
            backbone=config.backbone,
            pretrained=config.pretrained,
            feature_dim=config.feature_dim
        )
        
        self.nam_head = NAMHead(
            feature_dim=config.feature_dim,
            num_classes=config.num_classes,
            hidden_units=config.nam_hidden_units,
            dropout=config.dropout,
            use_exu=config.use_exu
        )
        
        logger.info("CentralizedNAM initialized (privacy baseline)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input images (batch_size, 3, height, width)
            
        Returns:
            Predictions (batch_size, num_classes)
        """
        features = self.feature_extractor(x)
        predictions, _ = self.nam_head(features)
        return predictions
    
    def forward_with_contributions(self, x: torch.Tensor):
        """Forward pass with contributions."""
        features = self.feature_extractor(x)
        predictions, contributions = self.nam_head(features)
        return predictions, features, contributions
    
    def get_explanations(self, x: torch.Tensor) -> np.ndarray:
        """Get NAM feature contributions as explanations.
        
        Args:
            x: Input images (batch_size, 3, height, width)
            
        Returns:
            Feature contributions (batch_size, feature_dim)
        """
        self.eval()
        with torch.no_grad():
            features = self.feature_extractor(x)
            contributions = self.nam_head.get_feature_contributions(features)
            
            # Stack contributions
            contributions = torch.cat(contributions, dim=1)
            
        return contributions.cpu().numpy()
