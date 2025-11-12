"""Data module for FedNAMs+ system.

This module handles dataset downloading, preprocessing, and federated partitioning.
"""

from .downloader import DataDownloader
from .preprocessor import DataPreprocessor, MIMICCXRDataset
from .partitioner import FederatedDataPartitioner
from .base_dataset import BaseDataset, DatasetRegistry

__all__ = [
    'DataDownloader',
    'DataPreprocessor',
    'MIMICCXRDataset',
    'FederatedDataPartitioner',
    'BaseDataset',
    'DatasetRegistry'
]
