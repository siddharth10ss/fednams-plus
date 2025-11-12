"""NIH Chest X-ray Dataset implementation."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from torchvision import transforms

from .base_dataset import BaseDataset, DatasetRegistry
from utils.exceptions import DataError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@DatasetRegistry.register('nih-cxr')
class NIHChestXrayDataset(BaseDataset):
    """NIH Chest X-ray Dataset (ChestX-ray8).
    
    Dataset contains 112,120 frontal-view X-ray images (1024x1024) from 30,805 unique patients.
    15 classes: 14 diseases + "No Finding"
    Labels extracted via NLP from radiological reports (>90% accuracy).
    
    Source: https://www.kaggle.com/datasets/nih-chest-xrays/data
    Paper: Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and 
           Benchmarks on Weakly-Supervised Classification and Localization of 
           Common Thorax Diseases." IEEE CVPR 2017
    """
    
    def __init__(self, data_dir: Path, transform: transforms.Compose = None):
        """Initialize NIH dataset.
        
        Args:
            data_dir: Root directory containing the dataset
            transform: Optional transforms to apply to images
        """
        super().__init__(data_dir)
        self.transform = transform
        
        # NIH pathology classes (15 total: 14 diseases + No Finding)
        # Based on ChestX-ray8 dataset from Wang et al.
        self.class_names = [
            'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
            'Edema', 'Emphysema', 'Fibrosis', 'Effusion',
            'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule',
            'Mass', 'Hernia', 'No Finding'
        ]
    
    def load(self) -> None:
        """Load NIH dataset from disk."""
        logger.info(f"Loading NIH Chest X-ray dataset from {self.data_dir}")
        
        # Load labels file
        labels_file = self.data_dir / 'Data_Entry_2017.csv'
        if not labels_file.exists():
            raise DataError(f"Labels file not found: {labels_file}")
        
        self.metadata = pd.read_csv(labels_file)
        logger.info(f"Loaded metadata for {len(self.metadata)} images")
        
        # Find all image directories
        image_dirs = []
        for img_dir in self.data_dir.glob('images*'):
            if img_dir.is_dir():
                image_dirs.append(img_dir)
        
        if not image_dirs:
            raise DataError(f"No image directories found in {self.data_dir}")
        
        logger.info(f"Found {len(image_dirs)} image directories")
        
        # Build image path mapping
        image_path_map = {}
        for img_dir in image_dirs:
            for img_file in img_dir.glob('*.png'):
                image_path_map[img_file.name] = img_file
        
        logger.info(f"Found {len(image_path_map)} total images")
        
        # Match images with labels
        self.image_paths = []
        valid_indices = []
        
        for idx, row in self.metadata.iterrows():
            img_name = row['Image Index']
            if img_name in image_path_map:
                self.image_paths.append(image_path_map[img_name])
                valid_indices.append(idx)
        
        # Filter metadata to only valid images
        self.metadata = self.metadata.iloc[valid_indices].reset_index(drop=True)
        
        logger.info(f"Matched {len(self.image_paths)} images with labels")
        
        # Process labels
        self.labels = self._process_labels()
        
        logger.info("Dataset loaded successfully")
    
    def _process_labels(self) -> np.ndarray:
        """Process labels from metadata.
        
        Returns:
            Binary label matrix (n_samples, n_classes)
        """
        n_samples = len(self.metadata)
        n_classes = len(self.class_names)
        labels_matrix = np.zeros((n_samples, n_classes), dtype=np.float32)
        
        for idx, row in self.metadata.iterrows():
            findings = str(row['Finding Labels']).split('|')
            for finding in findings:
                finding = finding.strip()
                if finding in self.class_names:
                    class_idx = self.class_names.index(finding)
                    labels_matrix[idx, class_idx] = 1.0
        
        return labels_matrix
    
    def preprocess(self, **kwargs) -> None:
        """Preprocess the dataset.
        
        For NIH dataset, preprocessing is minimal as images are already in PNG format.
        """
        logger.info("NIH dataset preprocessing (no additional steps required)")
        pass
    
    def get_labels(self) -> np.ndarray:
        """Get label array.
        
        Returns:
            Label array (n_samples, n_classes)
        """
        return self.labels
    
    def get_label_names(self) -> List[str]:
        """Get names of label classes.
        
        Returns:
            List of class names
        """
        return self.class_names
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return img, label
    
    def get_statistics(self) -> dict:
        """Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = super().get_statistics()
        
        # Add NIH-specific statistics
        stats.update({
            'class_names': self.class_names,
            'samples_per_class': self.labels.sum(axis=0).tolist(),
            'avg_labels_per_image': float(self.labels.sum(axis=1).mean()),
            'images_with_no_finding': int((self.labels[:, -1] == 1).sum())
        })
        
        return stats
