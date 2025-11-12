"""Feature extractor using CNN backbones for chest X-ray images."""

import torch
import torch.nn as nn
import torchvision.models as models

from utils.exceptions import ModelError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class FeatureExtractor(nn.Module):
    """Extracts feature representations from images using CNN backbones.
    
    Supports multiple pretrained CNN architectures for transfer learning.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        feature_dim: int = 512
    ):
        """Initialize the feature extractor.
        
        Args:
            backbone: CNN architecture ('resnet18', 'resnet50', 'densenet121')
            pretrained: Whether to use ImageNet pretrained weights
            feature_dim: Dimension of output feature vectors
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.feature_dim = feature_dim
        
        # Load backbone
        self.backbone = self._load_backbone()
        
        # Get the actual output dimension from backbone
        self.actual_feature_dim = self._get_backbone_output_dim()
        
        # Add projection layer if needed
        if self.actual_feature_dim != feature_dim:
            self.projection = nn.Linear(self.actual_feature_dim, feature_dim)
        else:
            self.projection = nn.Identity()
        
        logger.info(
            f"FeatureExtractor initialized: {backbone}, "
            f"pretrained={pretrained}, feature_dim={feature_dim}"
        )
    
    def _load_backbone(self) -> nn.Module:
        """Load the CNN backbone.
        
        Returns:
            CNN backbone model
            
        Raises:
            ModelError: If backbone is not supported
        """
        if self.backbone_name == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained)
            # Remove final FC layer
            backbone = nn.Sequential(*list(model.children())[:-1])
            
        elif self.backbone_name == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained)
            backbone = nn.Sequential(*list(model.children())[:-1])
            
        elif self.backbone_name == 'densenet121':
            model = models.densenet121(pretrained=self.pretrained)
            # DenseNet has different structure
            backbone = nn.Sequential(
                model.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            
        else:
            raise ModelError(
                f"Unsupported backbone: {self.backbone_name}. "
                f"Supported: resnet18, resnet50, densenet121"
            )
        
        return backbone
    
    def _get_backbone_output_dim(self) -> int:
        """Get the output dimension of the backbone.
        
        Returns:
            Output feature dimension
        """
        # Test with dummy input
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output = self.backbone(dummy_input)
            output = output.view(output.size(0), -1)
            return output.size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images.
        
        Args:
            x: Input images (batch_size, 3, height, width)
            
        Returns:
            Feature vectors (batch_size, feature_dim)
        """
        # Extract features
        features = self.backbone(x)
        
        # Flatten
        features = features.view(features.size(0), -1)
        
        # Project to desired dimension
        features = self.projection(features)
        
        return features
    
    def freeze_backbone(self):
        """Freeze backbone parameters for faster training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")
    
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
