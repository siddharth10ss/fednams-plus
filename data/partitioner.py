"""Federated data partitioner for creating non-IID client datasets."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import torch
from torch.utils.data import Dataset, Subset

from utils.exceptions import DataError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class FederatedDataPartitioner:
    """Partitions data into non-IID client subsets for federated learning.
    
    Supports multiple partitioning strategies:
    - Dirichlet: Label distribution heterogeneity using Dirichlet distribution
    - Pathology-based: Clients specialize in different pathologies
    - Quantity-based: Varying dataset sizes across clients
    """
    
    def __init__(self, num_clients: int, strategy: str = 'dirichlet', **kwargs):
        """Initialize the partitioner.
        
        Args:
            num_clients: Number of federated clients
            strategy: Partitioning strategy ('dirichlet', 'pathology', 'quantity')
            **kwargs: Strategy-specific parameters
        """
        self.num_clients = num_clients
        self.strategy = strategy
        self.kwargs = kwargs
        
        # Minimum samples per client
        self.min_samples = kwargs.get('min_samples', 100)
        
        logger.info(f"Initialized partitioner: {num_clients} clients, strategy={strategy}")
    
    def partition(self, dataset: Dataset) -> Dict[int, Subset]:
        """Partition dataset into client subsets.
        
        Args:
            dataset: Complete dataset to partition
            
        Returns:
            Dictionary mapping client_id to Subset
            
        Raises:
            DataError: If partitioning fails
        """
        if self.strategy == 'dirichlet':
            return self._partition_dirichlet(dataset)
        elif self.strategy == 'pathology':
            return self._partition_pathology(dataset)
        elif self.strategy == 'quantity':
            return self._partition_quantity(dataset)
        else:
            raise DataError(f"Unknown partitioning strategy: {self.strategy}")
    
    def _partition_dirichlet(self, dataset: Dataset) -> Dict[int, Subset]:
        """Partition using Dirichlet distribution for label heterogeneity.
        
        Args:
            dataset: Complete dataset
            
        Returns:
            Dictionary of client subsets
        """
        alpha = self.kwargs.get('alpha', 0.5)
        logger.info(f"Partitioning with Dirichlet distribution (alpha={alpha})")
        
        # Get all labels
        labels = self._extract_labels(dataset)
        n_samples = len(labels)
        n_classes = labels.shape[1]
        
        # For multi-label, use primary label (most common positive label)
        primary_labels = np.argmax(labels, axis=1)
        
        # Initialize client indices
        client_indices = [[] for _ in range(self.num_clients)]
        
        # Partition each class using Dirichlet distribution
        for class_idx in range(n_classes):
            # Get indices for this class
            class_indices = np.where(primary_labels == class_idx)[0]
            
            if len(class_indices) == 0:
                continue
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * self.num_clients)
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            
            # Split indices according to proportions
            split_indices = np.split(class_indices, proportions)
            
            # Assign to clients
            for client_id, indices in enumerate(split_indices):
                client_indices[client_id].extend(indices.tolist())
        
        # Ensure minimum samples per client
        client_indices = self._ensure_minimum_samples(client_indices, n_samples)
        
        # Create Subsets
        partitions = {}
        for client_id, indices in enumerate(client_indices):
            partitions[client_id] = Subset(dataset, indices)
            logger.info(f"Client {client_id}: {len(indices)} samples")
        
        return partitions
    
    def _partition_pathology(self, dataset: Dataset) -> Dict[int, Subset]:
        """Partition based on pathology specialization.
        
        Each client specializes in certain pathologies.
        
        Args:
            dataset: Complete dataset
            
        Returns:
            Dictionary of client subsets
        """
        logger.info("Partitioning with pathology-based strategy")
        
        labels = self._extract_labels(dataset)
        n_samples = len(labels)
        n_classes = labels.shape[1]
        
        # Assign pathologies to clients
        pathologies_per_client = max(1, n_classes // self.num_clients)
        
        client_indices = [[] for _ in range(self.num_clients)]
        
        for client_id in range(self.num_clients):
            # Assign specific pathologies to this client
            start_class = (client_id * pathologies_per_client) % n_classes
            end_class = min(start_class + pathologies_per_client, n_classes)
            client_pathologies = list(range(start_class, end_class))
            
            # Get samples with these pathologies
            for sample_idx in range(n_samples):
                if any(labels[sample_idx, p] == 1 for p in client_pathologies):
                    client_indices[client_id].append(sample_idx)
        
        # Ensure minimum samples
        client_indices = self._ensure_minimum_samples(client_indices, n_samples)
        
        # Create Subsets
        partitions = {}
        for client_id, indices in enumerate(client_indices):
            partitions[client_id] = Subset(dataset, indices)
            logger.info(f"Client {client_id}: {len(indices)} samples")
        
        return partitions
    
    def _partition_quantity(self, dataset: Dataset) -> Dict[int, Subset]:
        """Partition with varying dataset sizes across clients.
        
        Args:
            dataset: Complete dataset
            
        Returns:
            Dictionary of client subsets
        """
        logger.info("Partitioning with quantity-based strategy")
        
        n_samples = len(dataset)
        
        # Generate random proportions
        proportions = np.random.exponential(scale=1.0, size=self.num_clients)
        proportions = proportions / proportions.sum()
        
        # Calculate client sizes
        client_sizes = (proportions * n_samples).astype(int)
        
        # Ensure minimum samples
        for i in range(self.num_clients):
            if client_sizes[i] < self.min_samples:
                client_sizes[i] = self.min_samples
        
        # Adjust to match total
        diff = n_samples - client_sizes.sum()
        client_sizes[0] += diff
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        
        # Split indices
        client_indices = []
        start_idx = 0
        for size in client_sizes:
            end_idx = start_idx + size
            client_indices.append(indices[start_idx:end_idx].tolist())
            start_idx = end_idx
        
        # Create Subsets
        partitions = {}
        for client_id, indices in enumerate(client_indices):
            partitions[client_id] = Subset(dataset, indices)
            logger.info(f"Client {client_id}: {len(indices)} samples")
        
        return partitions
    
    def _ensure_minimum_samples(
        self,
        client_indices: List[List[int]],
        total_samples: int
    ) -> List[List[int]]:
        """Ensure each client has minimum number of samples.
        
        Args:
            client_indices: List of index lists for each client
            total_samples: Total number of samples available
            
        Returns:
            Adjusted client indices
        """
        for client_id in range(self.num_clients):
            if len(client_indices[client_id]) < self.min_samples:
                # Need more samples
                needed = self.min_samples - len(client_indices[client_id])
                
                # Find clients with excess samples
                available_indices = []
                for other_id in range(self.num_clients):
                    if other_id != client_id and len(client_indices[other_id]) > self.min_samples:
                        excess = len(client_indices[other_id]) - self.min_samples
                        available_indices.extend(client_indices[other_id][:excess])
                
                # Transfer samples
                if len(available_indices) >= needed:
                    transfer = np.random.choice(available_indices, needed, replace=False)
                    client_indices[client_id].extend(transfer.tolist())
                    
                    # Remove from other clients
                    for other_id in range(self.num_clients):
                        if other_id != client_id:
                            client_indices[other_id] = [
                                idx for idx in client_indices[other_id]
                                if idx not in transfer
                            ]
        
        return client_indices
    
    def _extract_labels(self, dataset: Dataset) -> np.ndarray:
        """Extract labels from dataset.
        
        Args:
            dataset: Dataset to extract labels from
            
        Returns:
            Label array (n_samples, n_classes)
        """
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label.numpy())
        
        return np.array(labels)
    
    def generate_statistics(
        self,
        partitions: Dict[int, Subset],
        output_dir: Path
    ) -> pd.DataFrame:
        """Generate statistics for partitioned data.
        
        Args:
            partitions: Dictionary of client subsets
            output_dir: Directory to save statistics
            
        Returns:
            DataFrame with partition statistics
        """
        logger.info("Generating partition statistics")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = []
        
        for client_id, subset in partitions.items():
            # Extract labels for this client
            labels = []
            for idx in subset.indices:
                _, label = subset.dataset[idx]
                labels.append(label.numpy())
            
            labels = np.array(labels)
            
            # Compute statistics (convert all numpy types to Python types)
            client_stats = {
                'client_id': int(client_id),
                'num_samples': int(len(subset)),
                'label_distribution': [float(x) for x in labels.sum(axis=0)],
                'avg_labels_per_sample': float(labels.sum(axis=1).mean()),
            }
            
            stats.append(client_stats)
            
            # Save client-specific metadata
            client_file = output_dir / f'client_{client_id}_metadata.json'
            with open(client_file, 'w') as f:
                json.dump(client_stats, f, indent=2)
        
        # Create DataFrame
        stats_df = pd.DataFrame(stats)
        
        # Save overall statistics
        stats_file = output_dir / 'partition_statistics.csv'
        stats_df.to_csv(stats_file, index=False)
        
        logger.info(f"Statistics saved to {output_dir}")
        
        return stats_df
