"""Federated orchestrator for coordinating training across clients."""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from torch.utils.data import DataLoader

from .local_trainer import LocalTrainer
from .aggregator import FedAvgAggregator
from configs.config import FedConfig
from utils.exceptions import TrainingError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class FederatedOrchestrator:
    """Coordinates federated training across multiple clients.
    
    Manages the complete federated learning workflow including client
    sampling, local training, aggregation, and global model distribution.
    """
    
    def __init__(
        self,
        clients: List[LocalTrainer],
        aggregator: FedAvgAggregator,
        config: FedConfig
    ):
        """Initialize federated orchestrator.
        
        Args:
            clients: List of local trainers (one per client)
            aggregator: Federated aggregator
            config: Federated learning configuration
        """
        self.clients = clients
        self.aggregator = aggregator
        self.config = config
        
        self.current_round = 0
        self.global_metrics = []
        self.client_metrics = {i: [] for i in range(len(clients))}
        
        # Communication cost tracking
        self.total_params_transmitted = 0
        
        logger.info(
            f"FederatedOrchestrator initialized: "
            f"{len(clients)} clients, {config.num_rounds} rounds"
        )
    
    def run_round(
        self,
        round_num: int,
        train_loaders: List[DataLoader],
        val_loaders: Optional[List[DataLoader]] = None
    ) -> Dict[str, Any]:
        """Execute one federated training round.
        
        Args:
            round_num: Current round number
            train_loaders: Training data loaders for each client
            val_loaders: Optional validation data loaders
            
        Returns:
            Dictionary of round metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_num}/{self.config.num_rounds}")
        logger.info(f"{'='*60}")
        
        # Sample clients for this round
        selected_clients = self._sample_clients()
        logger.info(f"Selected {len(selected_clients)} clients: {selected_clients}")
        
        # Local training on selected clients
        client_params = []
        client_weights = []
        round_metrics = {'clients': {}}
        
        for client_id in selected_clients:
            logger.info(f"\nTraining Client {client_id}...")
            
            trainer = self.clients[client_id]
            train_loader = train_loaders[client_id]
            
            # Train for local epochs
            for epoch in range(self.config.num_local_epochs):
                train_metrics = trainer.train_epoch(train_loader)
                
                if epoch == self.config.num_local_epochs - 1:
                    logger.info(
                        f"  Epoch {epoch+1}: "
                        f"loss={train_metrics['loss']:.4f}, "
                        f"lr={train_metrics['lr']:.6f}"
                    )
            
            # Evaluate if validation data provided
            if val_loaders is not None:
                val_metrics = trainer.evaluate(val_loaders[client_id])
                logger.info(
                    f"  Validation: "
                    f"loss={val_metrics['loss']:.4f}, "
                    f"acc={val_metrics['accuracy']:.4f}"
                )
                train_metrics.update(val_metrics)
            
            # Store metrics
            round_metrics['clients'][client_id] = train_metrics
            self.client_metrics[client_id].append(train_metrics)
            
            # Get model parameters
            params = trainer.get_model_parameters()
            client_params.append(params)
            
            # Weight by dataset size
            client_weights.append(len(train_loader.dataset))
        
        # Aggregate parameters
        logger.info(f"\nAggregating parameters from {len(selected_clients)} clients...")
        
        # Set aggregation weights based on dataset sizes
        self.aggregator.set_aggregation_weights(client_weights)
        
        # Perform aggregation
        global_params = self.aggregator.aggregate(client_params)
        
        # Track communication cost
        num_params = sum(p.numel() for p in global_params.values())
        self.total_params_transmitted += num_params * len(selected_clients) * 2  # up + down
        
        # Distribute global model to all clients
        logger.info("Distributing global model to all clients...")
        for trainer in self.clients:
            trainer.set_model_parameters(global_params)
        
        # Compute global metrics (average across selected clients)
        global_train_loss = np.mean([
            round_metrics['clients'][cid]['loss']
            for cid in selected_clients
        ])
        
        round_metrics['global'] = {
            'train_loss': global_train_loss,
            'selected_clients': selected_clients,
            'num_params': num_params,
            'total_params_transmitted': self.total_params_transmitted
        }
        
        if val_loaders is not None:
            global_val_loss = np.mean([
                round_metrics['clients'][cid]['loss']
                for cid in selected_clients
                if 'loss' in round_metrics['clients'][cid]
            ])
            global_val_acc = np.mean([
                round_metrics['clients'][cid]['accuracy']
                for cid in selected_clients
                if 'accuracy' in round_metrics['clients'][cid]
            ])
            round_metrics['global']['val_loss'] = global_val_loss
            round_metrics['global']['val_accuracy'] = global_val_acc
        
        self.global_metrics.append(round_metrics['global'])
        self.current_round = round_num
        
        logger.info(f"\nRound {round_num} Summary:")
        logger.info(f"  Global train loss: {global_train_loss:.4f}")
        if val_loaders is not None:
            logger.info(f"  Global val loss: {global_val_loss:.4f}")
            logger.info(f"  Global val accuracy: {global_val_acc:.4f}")
        
        return round_metrics
    
    def run_training(
        self,
        train_loaders: List[DataLoader],
        val_loaders: Optional[List[DataLoader]] = None
    ) -> Dict[str, Any]:
        """Run complete federated training.
        
        Args:
            train_loaders: Training data loaders for each client
            val_loaders: Optional validation data loaders
            
        Returns:
            Dictionary of training results
        """
        logger.info(f"\nStarting federated training: {self.config.num_rounds} rounds")
        
        for round_num in range(1, self.config.num_rounds + 1):
            round_metrics = self.run_round(round_num, train_loaders, val_loaders)
            
            # Check for convergence (optional)
            if self._check_convergence():
                logger.info(f"Convergence detected at round {round_num}")
                break
        
        logger.info(f"\nFederated training complete!")
        
        # Compile results
        results = {
            'global_metrics': self.global_metrics,
            'client_metrics': self.client_metrics,
            'total_rounds': self.current_round,
            'communication_cost': {
                'total_params_transmitted': self.total_params_transmitted,
                'params_per_round': self.total_params_transmitted / self.current_round,
                'mb_transmitted': self.total_params_transmitted * 4 / (1024**2)  # float32
            }
        }
        
        return results
    
    def _sample_clients(self) -> List[int]:
        """Sample clients for current round.
        
        Returns:
            List of selected client indices
        """
        num_clients = len(self.clients)
        num_selected = max(
            self.config.min_clients,
            int(num_clients * self.config.client_fraction)
        )
        
        # Random sampling
        selected = np.random.choice(
            num_clients,
            size=min(num_selected, num_clients),
            replace=False
        )
        
        return sorted(selected.tolist())
    
    def _check_convergence(self) -> bool:
        """Check if training has converged.
        
        Returns:
            True if converged, False otherwise
        """
        if len(self.global_metrics) < 10:
            return False
        
        # Check if loss has plateaued
        recent_losses = [m['train_loss'] for m in self.global_metrics[-10:]]
        loss_std = np.std(recent_losses)
        
        if loss_std < 0.001:
            return True
        
        return False
    
    def save_checkpoint(self, round_num: int, path: Path):
        """Save federated training checkpoint.
        
        Args:
            round_num: Current round number
            path: Path to save checkpoint
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'round': round_num,
            'global_metrics': self.global_metrics,
            'client_metrics': self.client_metrics,
            'total_params_transmitted': self.total_params_transmitted,
            'client_states': [
                {
                    'model_state': client.model.state_dict(),
                    'optimizer_state': client.optimizer.state_dict()
                }
                for client in self.clients
            ]
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path) -> int:
        """Load federated training checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Round number to resume from
        """
        checkpoint = torch.load(path)
        
        self.current_round = checkpoint['round']
        self.global_metrics = checkpoint['global_metrics']
        self.client_metrics = checkpoint['client_metrics']
        self.total_params_transmitted = checkpoint['total_params_transmitted']
        
        # Restore client states
        for client, state in zip(self.clients, checkpoint['client_states']):
            client.model.load_state_dict(state['model_state'])
            client.optimizer.load_state_dict(state['optimizer_state'])
        
        logger.info(f"Checkpoint loaded from {path}, resuming from round {self.current_round}")
        
        return self.current_round
