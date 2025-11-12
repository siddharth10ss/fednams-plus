"""Abstract base dataset interface for extensibility."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Abstract base class for medical imaging datasets.
    
    Provides a common interface for different datasets (MIMIC-CXR, CheXpert, NIH, etc.)
    to enable easy switching and extension.
    """
    
    def __init__(self, data_dir: Path):
        """Initialize the dataset.
        
        Args:
            data_dir: Root directory containing the dataset
        """
        self.data_dir = Path(data_dir)
        self.image_paths: List[Path] = []
        self.labels: np.ndarray = np.array([])
        self.metadata: pd.DataFrame = pd.DataFrame()
    
    @abstractmethod
    def load(self) -> None:
        """Load dataset from disk.
        
        This method should:
        - Discover image files
        - Load label information
        - Load metadata
        - Populate self.image_paths, self.labels, self.metadata
        """
        pass
    
    @abstractmethod
    def preprocess(self, **kwargs) -> None:
        """Preprocess the dataset.
        
        This method should handle dataset-specific preprocessing:
        - Image format conversion
        - Label encoding
        - Data cleaning
        """
        pass
    
    @abstractmethod
    def get_labels(self) -> np.ndarray:
        """Get label array.
        
        Returns:
            Label array (n_samples, n_classes)
        """
        pass
    
    @abstractmethod
    def get_label_names(self) -> List[str]:
        """Get names of label classes.
        
        Returns:
            List of class names
        """
        pass
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, label)
        """
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'num_samples': len(self),
            'num_classes': self.labels.shape[1] if len(self.labels.shape) > 1 else 1,
            'label_distribution': self.labels.sum(axis=0).tolist() if len(self.labels) > 0 else [],
            'data_dir': str(self.data_dir)
        }


class DatasetRegistry:
    """Registry for managing multiple dataset implementations.
    
    Allows easy registration and retrieval of dataset classes.
    """
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a dataset class.
        
        Args:
            name: Name to register the dataset under
            
        Example:
            @DatasetRegistry.register('mimic-cxr')
            class MIMICCXRDataset(BaseDataset):
                ...
        """
        def decorator(dataset_class):
            cls._registry[name] = dataset_class
            return dataset_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> type:
        """Get a registered dataset class.
        
        Args:
            name: Name of the dataset
            
        Returns:
            Dataset class
            
        Raises:
            KeyError: If dataset not found
        """
        if name not in cls._registry:
            raise KeyError(
                f"Dataset '{name}' not found. "
                f"Available datasets: {list(cls._registry.keys())}"
            )
        return cls._registry[name]
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered datasets.
        
        Returns:
            List of dataset names
        """
        return list(cls._registry.keys())
