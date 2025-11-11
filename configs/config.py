"""Configuration dataclasses for FedNAMs+ system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing.
    
    Attributes:
        image_size: Target size for image resizing (height, width)
        normalization: Normalization strategy ('imagenet' or 'custom')
        augmentation: Whether to apply data augmentation
        augmentation_params: Parameters for augmentation transforms
    """
    image_size: Tuple[int, int] = (224, 224)
    normalization: str = "imagenet"
    augmentation: bool = True
    augmentation_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for model architecture.
    
    Attributes:
        backbone: CNN backbone architecture ('resnet18', 'resnet50', 'densenet121')
        pretrained: Whether to use pretrained ImageNet weights
        feature_dim: Dimension of extracted feature vectors
        num_classes: Number of output classes
        nam_hidden_units: Hidden layer sizes for NAM sub-networks
        dropout: Dropout probability for regularization
        use_exu: Whether to use ExU activation for monotonicity
    """
    backbone: str = "resnet18"
    pretrained: bool = True
    feature_dim: int = 512
    num_classes: int = 14
    nam_hidden_units: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.3
    use_exu: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training process.
    
    Attributes:
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        num_local_epochs: Number of epochs per client per round
        optimizer: Optimizer type ('adam', 'sgd', 'adamw')
        scheduler: Learning rate scheduler ('cosine', 'step', None)
        early_stopping_patience: Patience for early stopping
        mixed_precision: Whether to use mixed precision training
    """
    batch_size: int = 32
    learning_rate: float = 0.001
    num_local_epochs: int = 5
    optimizer: str = "adam"
    scheduler: Optional[str] = "cosine"
    early_stopping_patience: int = 10
    mixed_precision: bool = True


@dataclass
class FedConfig:
    """Configuration for federated learning.
    
    Attributes:
        num_clients: Number of federated clients
        num_rounds: Number of federated training rounds
        client_fraction: Fraction of clients to sample per round
        aggregation_weights: Optional weights for aggregation (None for uniform)
        min_clients: Minimum number of clients required per round
    """
    num_clients: int = 5
    num_rounds: int = 100
    client_fraction: float = 1.0
    aggregation_weights: Optional[List[float]] = None
    min_clients: int = 3


@dataclass
class ExperimentConfig:
    """Configuration for complete experiment.
    
    Attributes:
        experiment_name: Name of the experiment
        data_config: Data preprocessing configuration
        model_config: Model architecture configuration
        training_config: Training process configuration
        fed_config: Federated learning configuration
        output_dir: Directory for saving outputs
        seed: Random seed for reproducibility
        device: Device for computation ('cuda' or 'cpu')
    """
    experiment_name: str
    data_config: PreprocessConfig
    model_config: ModelConfig
    training_config: TrainingConfig
    fed_config: FedConfig
    output_dir: Path
    seed: int = 42
    device: str = "cuda"
