"""Configuration loader for YAML files."""

import yaml
from pathlib import Path
from typing import Dict, Any

from .config import (
    PreprocessConfig,
    ModelConfig,
    TrainingConfig,
    FedConfig,
    ExperimentConfig
)


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


class ConfigLoader:
    """Loads and validates configuration from YAML files."""
    
    @staticmethod
    def load_yaml(config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file.
        
        Args:
            config_path: Path to YAML file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationError: If file not found or invalid YAML
        """
        if not config_path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_path}: {str(e)}")
    
    @staticmethod
    def create_experiment_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Create ExperimentConfig from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ExperimentConfig instance
            
        Raises:
            ConfigurationError: If required keys are missing
        """
        try:
            # Data config
            image_size = config_dict['data']['image_size']
            # Handle both single int and list/tuple formats
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            else:
                image_size = tuple(image_size)
            
            data_config = PreprocessConfig(
                image_size=image_size,
                normalization=config_dict['data']['normalization'],
                augmentation=config_dict['data']['augmentation'],
                augmentation_params=config_dict['data'].get('augmentation_params', {})
            )
            
            # Model config
            model_config = ModelConfig(
                backbone=config_dict['model']['backbone'],
                pretrained=config_dict['model']['pretrained'],
                feature_dim=config_dict['model']['feature_dim'],
                num_classes=config_dict['model']['num_classes'],
                nam_hidden_units=config_dict['model']['nam_hidden_units'],
                dropout=config_dict['model']['dropout'],
                use_exu=config_dict['model']['use_exu']
            )
            
            # Training config
            training_config = TrainingConfig(
                batch_size=config_dict['training']['batch_size'],
                learning_rate=config_dict['training']['learning_rate'],
                num_local_epochs=config_dict['training']['num_local_epochs'],
                optimizer=config_dict['training']['optimizer'],
                scheduler=config_dict['training'].get('scheduler'),
                early_stopping_patience=config_dict['training']['early_stopping_patience'],
                mixed_precision=config_dict['training']['mixed_precision']
            )
            
            # Fed config
            fed_config = FedConfig(
                num_clients=config_dict['federated']['num_clients'],
                num_rounds=config_dict['federated']['num_rounds'],
                client_fraction=config_dict['federated']['client_fraction'],
                min_clients=config_dict['federated']['min_clients']
            )
            
            # Experiment config
            experiment_config = ExperimentConfig(
                experiment_name=config_dict['experiment_name'],
                data_config=data_config,
                model_config=model_config,
                training_config=training_config,
                fed_config=fed_config,
                output_dir=Path(config_dict.get('output_dir', 'outputs')),
                seed=config_dict['seed'],
                device=config_dict['device']
            )
            
            return experiment_config
            
        except KeyError as e:
            raise ConfigurationError(f"Missing required config key: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Error creating config: {str(e)}")
    
    @staticmethod
    def load_experiment_config(config_path: Path) -> ExperimentConfig:
        """Load experiment configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            ExperimentConfig instance
        """
        config_dict = ConfigLoader.load_yaml(config_path)
        return ConfigLoader.create_experiment_config(config_dict)
