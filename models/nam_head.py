"""Neural Additive Model (NAM) head for interpretable predictions."""

import torch
import torch.nn as nn
from typing import List, Tuple

from utils.exceptions import ModelError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ExU(nn.Module):
    """Exponential Unit (ExU) activation for monotonic functions.
    
    ExU(x) = exp(w) * (exp(x) - 1) for x >= 0
    ExU(x) = exp(w) * (exp(x) - exp(-x)) for x < 0
    """
    
    def __init__(self, in_features: int):
        """Initialize ExU activation.
        
        Args:
            in_features: Number of input features
        """
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ExU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        return torch.exp(self.weights) * (torch.exp(x) - 1)


class FeatureNN(nn.Module):
    """Neural network for a single feature in NAM.
    
    Each feature has its own small neural network that learns
    its contribution to the final prediction.
    """
    
    def __init__(
        self,
        hidden_units: List[int],
        dropout: float = 0.3,
        use_exu: bool = False
    ):
        """Initialize feature neural network.
        
        Args:
            hidden_units: List of hidden layer sizes
            dropout: Dropout probability
            use_exu: Whether to use ExU activation
        """
        super().__init__()
        
        layers = []
        in_dim = 1  # Each feature NN takes 1 input
        
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            if use_exu:
                layers.append(ExU(hidden_dim))
            else:
                layers.append(nn.ReLU())
            
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Final layer outputs single value (feature contribution)
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute feature contribution.
        
        Args:
            x: Feature value (batch_size, 1)
            
        Returns:
            Feature contribution (batch_size, 1)
        """
        return self.network(x)


class NAMHead(nn.Module):
    """Neural Additive Model head for interpretable predictions.
    
    NAM learns separate neural networks for each feature dimension,
    then combines their outputs additively for the final prediction.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_units: List[int] = [64, 32],
        dropout: float = 0.3,
        use_exu: bool = False
    ):
        """Initialize NAM head.
        
        Args:
            feature_dim: Dimension of input features
            num_classes: Number of output classes
            hidden_units: Hidden layer sizes for each feature NN
            dropout: Dropout probability
            use_exu: Whether to use ExU activation for monotonicity
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.use_exu = use_exu
        
        # Create separate neural network for each feature
        self.feature_nns = nn.ModuleList([
            FeatureNN(hidden_units, dropout, use_exu)
            for _ in range(feature_dim)
        ])
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(num_classes))
        
        # Final linear layer to map to num_classes
        self.output_layer = nn.Linear(feature_dim, num_classes)
        
        logger.info(
            f"NAMHead initialized: feature_dim={feature_dim}, "
            f"num_classes={num_classes}, hidden_units={hidden_units}"
        )
    
    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through NAM head.
        
        Args:
            features: Input features (batch_size, feature_dim)
            
        Returns:
            Tuple of:
                - predictions: Final predictions (batch_size, num_classes)
                - contributions: List of feature contributions
        """
        batch_size = features.size(0)
        
        # Compute contribution from each feature
        contributions = []
        for i, feature_nn in enumerate(self.feature_nns):
            # Extract single feature
            feature_value = features[:, i:i+1]  # (batch_size, 1)
            
            # Compute contribution
            contribution = feature_nn(feature_value)  # (batch_size, 1)
            contributions.append(contribution)
        
        # Stack contributions
        all_contributions = torch.cat(contributions, dim=1)  # (batch_size, feature_dim)
        
        # Map to output classes
        predictions = self.output_layer(all_contributions) + self.bias
        
        return predictions, contributions
    
    def get_feature_contributions(
        self,
        features: torch.Tensor
    ) -> List[torch.Tensor]:
        """Get individual feature contributions without final prediction.
        
        Args:
            features: Input features (batch_size, feature_dim)
            
        Returns:
            List of feature contributions
        """
        contributions = []
        for i, feature_nn in enumerate(self.feature_nns):
            feature_value = features[:, i:i+1]
            contribution = feature_nn(feature_value)
            contributions.append(contribution)
        
        return contributions
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())
