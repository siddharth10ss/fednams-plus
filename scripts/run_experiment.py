"""CLI script for running FedNAMs+ experiments."""

import argparse
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import (
    PreprocessConfig,
    ModelConfig,
    TrainingConfig,
    FedConfig,
    ExperimentConfig
)
from experiments import ExperimentRunner


def load_config(config_path: str) -> ExperimentConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        ExperimentConfig object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config objects
    data_config = PreprocessConfig(
        image_size=tuple(config_dict['data']['image_size']),
        normalization=config_dict['data']['normalization'],
        augmentation=config_dict['data']['augmentation'],
        augmentation_params=config_dict['data'].get('augmentation_params', {})
    )
    
    model_config = ModelConfig(
        backbone=config_dict['model']['backbone'],
        pretrained=config_dict['model']['pretrained'],
        feature_dim=config_dict['model']['feature_dim'],
        num_classes=config_dict['model']['num_classes'],
        nam_hidden_units=config_dict['model']['nam_hidden_units'],
        dropout=config_dict['model']['dropout'],
        use_exu=config_dict['model']['use_exu']
    )
    
    training_config = TrainingConfig(
        batch_size=config_dict['training']['batch_size'],
        learning_rate=config_dict['training']['learning_rate'],
        num_local_epochs=config_dict['training']['num_local_epochs'],
        optimizer=config_dict['training']['optimizer'],
        scheduler=config_dict['training'].get('scheduler'),
        early_stopping_patience=config_dict['training']['early_stopping_patience'],
        mixed_precision=config_dict['training']['mixed_precision']
    )
    
    fed_config = FedConfig(
        num_clients=config_dict['federated']['num_clients'],
        num_rounds=config_dict['federated']['num_rounds'],
        client_fraction=config_dict['federated']['client_fraction'],
        min_clients=config_dict['federated']['min_clients']
    )
    
    experiment_config = ExperimentConfig(
        experiment_name=config_dict['experiment_name'],
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        fed_config=fed_config,
        output_dir=Path(f"outputs/{config_dict['experiment_name']}"),
        seed=config_dict['seed'],
        device=config_dict['device']
    )
    
    return experiment_config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run FedNAMs+ experiments'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment.yaml',
        help='Path to experiment configuration file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu), overrides config'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed, overrides config'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override with CLI arguments
    if args.device:
        config.device = args.device
    if args.seed:
        config.seed = args.seed
    
    # Create experiment runner
    print(f"\nInitializing experiment: {config.experiment_name}")
    runner = ExperimentRunner(config)
    
    # Run experiment
    print("\nRunning experiment...")
    results = runner.run_experiment()
    
    # Save results
    print("\nSaving results...")
    runner.save_results(results)
    
    print(f"\n✓ Experiment complete!")
    print(f"Results saved to: {config.output_dir}")


if __name__ == '__main__':
    main()
