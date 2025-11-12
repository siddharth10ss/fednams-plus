"""Training module for FedNAMs+ system.

This module manages local training, federated aggregation, and training orchestration.
"""

from .local_trainer import LocalTrainer
from .aggregator import FedAvgAggregator
from .orchestrator import FederatedOrchestrator

__all__ = [
    'LocalTrainer',
    'FedAvgAggregator',
    'FederatedOrchestrator'
]
