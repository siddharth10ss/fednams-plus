"""Data module for FedNAMs+ system.

This module handles dataset downloading, preprocessing, and federated partitioning.
"""

from .downloader import DataDownloader
from .preprocessor import DataPreprocessor, MIMICCXRDataset
from .partitioner import FederatedDataPartitioner
from .base_dataset import BaseDataset, DatasetRegistry
from .nih_dataset import NIHChestXrayDataset

__all__ = [
    'DataDownloader',
    'DataPreprocessor',
    'MIMICCXRDataset',
    'NIHChestXrayDataset',
    'FederatedDataPartitioner',
    'BaseDataset',
    'DatasetRegistry'
]
