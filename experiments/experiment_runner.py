"""Experiment runner for orchestrating complete experimental workflows."""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from configs.config import ExperimentConfig
from .evaluation_metrics import EvaluationMetrics
from utils.logging_utils import setup_logger, get_logger

logger = get_logger(__name__)


class ExperimentRunner:
    """Orchestrates complete experimental workflows.
    
    Manages setup, execution, evaluation, and result saving for experiments.
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup experiment-specific logger
        log_file = self.output_dir / 'experiment.log'
        self.logger = setup_logger(
            f"experiment_{config.experiment_name}",
            log_file=log_file
        )
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        self.logger.info(f"ExperimentRunner initialized: {config.experiment_name}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility.
        
        Args:
            seed: Random seed
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.logger.info(f"Random seeds set to {seed}")
    
    def setup_experiment(self):
        """Setup experiment environment."""
        # Create subdirectories
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'results').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
        # Save configuration
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'experiment_name': self.config.experiment_name,
                'seed': self.config.seed,
                'device': self.config.device,
                'data_config': self.config.data_config.__dict__,
                'model_config': self.config.model_config.__dict__,
                'training_config': self.config.training_config.__dict__,
                'fed_config': self.config.fed_config.__dict__
            }, f, indent=2)
        
        self.logger.info("Experiment setup complete")
    
    def run_experiment(self, data_dir: Path) -> Dict[str, Any]:
        """Run complete experiment.
        
        Args:
            data_dir: Path to dataset directory
        
        Returns:
            Dictionary of experiment results
        """
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        
        # Setup
        self.setup_experiment()
        
        try:
            # 1. Load and prepare data
            self.logger.info("Loading dataset...")
            train_loader, val_loader, test_loader, client_loaders = self._load_data(data_dir)
            
            # 2. Create model
            self.logger.info("Creating model...")
            model = self._create_model()
            
            # 3. Run federated training
            self.logger.info("Starting federated training...")
            training_history = self._run_federated_training(model, client_loaders, val_loader)
            
            # 4. Evaluate on test set
            self.logger.info("Evaluating on test set...")
            test_metrics = self.evaluate_model(model, test_loader)
            
            # 5. Compile results
            results = {
                'experiment_name': self.config.experiment_name,
                'status': 'completed',
                'training_history': training_history,
                'test_metrics': test_metrics,
                'config': {
                    'num_clients': self.config.fed_config.num_clients,
                    'num_rounds': self.config.fed_config.num_rounds,
                    'batch_size': self.config.training_config.batch_size,
                    'learning_rate': self.config.training_config.learning_rate
                }
            }
            
            self.logger.info("Experiment completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}", exc_info=True)
            return {
                'experiment_name': self.config.experiment_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def _load_data(self, data_dir: Path):
        """Load and partition data for federated learning.
        
        Args:
            data_dir: Path to dataset directory
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader, client_loaders)
        """
        from data import NIHChestXrayDataset, DataPreprocessor, FederatedDataPartitioner
        
        # Load dataset
        dataset = NIHChestXrayDataset(data_dir=data_dir)
        dataset.load()
        
        self.logger.info(f"Loaded {len(dataset)} images")
        
        # Create preprocessor
        preprocessor = DataPreprocessor(self.config.data_config)
        
        # Apply transforms to dataset
        dataset.transform = preprocessor.train_transform
        
        # Split into train/val/test
        train_dataset, val_dataset, test_dataset = preprocessor.create_train_val_test_split(
            dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=self.config.seed
        )
        
        # Partition training data for federated learning
        partitioner = FederatedDataPartitioner(
            num_clients=self.config.fed_config.num_clients,
            strategy='dirichlet',
            alpha=0.5,
            min_samples=100
        )
        
        client_datasets = partitioner.partition(train_dataset)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training_config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training_config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training_config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Create client loaders
        client_loaders = {}
        for client_id, client_data in client_datasets.items():
            client_loaders[client_id] = DataLoader(
                client_data,
                batch_size=self.config.training_config.batch_size,
                shuffle=True,
                num_workers=2
            )
        
        self.logger.info(f"Created {len(client_loaders)} client data loaders")
        
        return train_loader, val_loader, test_loader, client_loaders
    
    def _create_model(self):
        """Create FedNAMs+ model.
        
        Returns:
            FedNAMs model instance
        """
        from models import FedNAMsModel
        
        # Pass the entire ModelConfig object
        model = FedNAMsModel(config=self.config.model_config)
        
        model = model.to(self.config.device)
        
        self.logger.info(f"Created {self.config.model_config.backbone} model with NAM head")
        
        return model
    
    def _run_federated_training(self, model, client_loaders, val_loader):
        """Run federated training across clients.
        
        Args:
            model: Model to train
            client_loaders: Dictionary of client data loaders
            val_loader: Validation data loader
            
        Returns:
            Training history dictionary
        """
        from training import LocalTrainer, FedAvgAggregator, FederatedOrchestrator
        
        # Create local trainers for each client
        clients = []
        for client_id in range(self.config.fed_config.num_clients):
            trainer = LocalTrainer(
                model=model,
                config=self.config.training_config,
                device=self.config.device
            )
            clients.append(trainer)
        
        # Create aggregator
        aggregator = FedAvgAggregator()
        
        # Create orchestrator
        orchestrator = FederatedOrchestrator(
            clients=clients,
            aggregator=aggregator,
            config=self.config.fed_config,
            num_local_epochs=self.config.training_config.num_local_epochs
        )
        
        # Run training
        # Note: orchestrator expects val_loaders (list), but we have one val_loader
        # We'll use the same val_loader for all clients
        val_loaders = [val_loader] * self.config.fed_config.num_clients
        
        history = orchestrator.run_training(
            train_loaders=list(client_loaders.values()),
            val_loaders=val_loaders
        )
        
        return history
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on test data.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        device = self.config.device
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate all batches
        y_pred = np.concatenate(all_preds, axis=0)
        y_prob = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
        
        # Compute metrics
        metrics = EvaluationMetrics.compute_classification_metrics(
            y_true, y_pred, y_prob
        )
        
        self.logger.info(f"Model evaluation complete: {len(y_true)} samples")
        
        return metrics
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: Optional[Path] = None
    ):
        """Save experiment results.
        
        Args:
            results: Results dictionary
            output_dir: Optional output directory (uses config default if None)
        """
        if output_dir is None:
            output_dir = self.output_dir / 'results'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        results_file = output_dir / 'results.json'
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = self._convert_to_json_serializable(results)
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
