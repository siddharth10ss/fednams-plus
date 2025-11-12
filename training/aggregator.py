"""Federated aggregation using FedAvg algorithm."""

import torch
from typing import Dict, List, Optional

from utils.exceptions import TrainingError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class FedAvgAggregator:
    """Aggregates client model parameters using Federated Averaging.
    
    Implements the FedAvg algorithm for combining model updates from
    multiple clients in federated learning.
    """
    
    def __init__(self, aggregation_weights: Optional[List[float]] = None):
        """Initialize FedAvg aggregator.
        
        Args:
            aggregation_weights: Optional weights for each client
                                If None, uses uniform weighting
        """
        self.aggregation_weights = aggregation_weights
        logger.info(
            f"FedAvgAggregator initialized: "
            f"weighted={aggregation_weights is not None}"
        )
    
    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client parameters using weighted averaging.
        
        Args:
            client_params: List of parameter dictionaries from clients
            
        Returns:
            Aggregated global parameters
            
        Raises:
            TrainingError: If aggregation fails
        """
        if not client_params:
            raise TrainingError("No client parameters provided for aggregation")
        
        num_clients = len(client_params)
        
        # Determine weights
        if self.aggregation_weights is None:
            # Uniform weighting
            weights = [1.0 / num_clients] * num_clients
        else:
            # Use provided weights
            if len(self.aggregation_weights) != num_clients:
                raise TrainingError(
                    f"Number of weights ({len(self.aggregation_weights)}) "
                    f"does not match number of clients ({num_clients})"
                )
            # Normalize weights
            total_weight = sum(self.aggregation_weights)
            weights = [w / total_weight for w in self.aggregation_weights]
        
        logger.info(f"Aggregating parameters from {num_clients} clients")
        
        # Initialize global parameters
        global_params = {}
        
        # Get parameter names from first client
        param_names = client_params[0].keys()
        
        # Aggregate each parameter
        for param_name in param_names:
            # Collect parameter from all clients
            param_list = []
            for client_param in client_params:
                if param_name not in client_param:
                    raise TrainingError(
                        f"Parameter {param_name} not found in client parameters"
                    )
                param_list.append(client_param[param_name])
            
            # Weighted average
            global_params[param_name] = self.weighted_average(param_list, weights)
        
        logger.info(f"Aggregation complete: {len(global_params)} parameters")
        
        return global_params
    
    def weighted_average(
        self,
        tensors: List[torch.Tensor],
        weights: List[float]
    ) -> torch.Tensor:
        """Compute weighted average of tensors.
        
        Args:
            tensors: List of tensors to average
            weights: Weight for each tensor
            
        Returns:
            Weighted average tensor
            
        Raises:
            TrainingError: If tensor shapes don't match
        """
        if len(tensors) != len(weights):
            raise TrainingError(
                f"Number of tensors ({len(tensors)}) "
                f"does not match number of weights ({len(weights)})"
            )
        
        # Check shapes match
        reference_shape = tensors[0].shape
        for i, tensor in enumerate(tensors[1:], 1):
            if tensor.shape != reference_shape:
                raise TrainingError(
                    f"Tensor shape mismatch: tensor 0 has shape {reference_shape}, "
                    f"tensor {i} has shape {tensor.shape}"
                )
        
        # Compute weighted sum
        result = torch.zeros_like(tensors[0])
        for tensor, weight in zip(tensors, weights):
            result += weight * tensor
        
        return result
    
    def set_aggregation_weights(self, weights: List[float]):
        """Update aggregation weights.
        
        Args:
            weights: New weights for aggregation
        """
        self.aggregation_weights = weights
        logger.info(f"Aggregation weights updated: {len(weights)} clients")
