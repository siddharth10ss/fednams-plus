"""Data preprocessor for MIMIC-CXR dataset."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms

from configs.config import PreprocessConfig
from utils.exceptions import DataError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocesses chest X-ray images and labels for model training.
    
    Handles:
    - Image resizing and normalization
    - Data augmentation
    - Multi-label encoding
    - Train/val/test splitting
    """
    
    def __init__(self, config: PreprocessConfig):
        """Initialize the preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.image_size = config.image_size
        
        # ImageNet normalization statistics
        if config.normalization == "imagenet":
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            # Custom normalization (can be computed from data)
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        
        # Define transforms
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()
    
    def _get_train_transform(self) -> transforms.Compose:
        """Get training data transforms with augmentation.
        
        Returns:
            Composed transforms for training
        """
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ]
        
        if self.config.augmentation:
            # Add augmentation before ToTensor
            aug_params = self.config.augmentation_params
            transform_list.insert(1, transforms.RandomHorizontalFlip(
                p=aug_params.get('horizontal_flip_prob', 0.5)
            ))
            transform_list.insert(2, transforms.RandomRotation(
                degrees=aug_params.get('rotation_degrees', 10)
            ))
            transform_list.insert(3, transforms.ColorJitter(
                brightness=aug_params.get('brightness', 0.2),
                contrast=aug_params.get('contrast', 0.2)
            ))
        
        # Add normalization at the end
        transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))
        
        return transforms.Compose(transform_list)
    
    def _get_test_transform(self) -> transforms.Compose:
        """Get test data transforms without augmentation.
        
        Returns:
            Composed transforms for testing
        """
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def preprocess_images(self, image_paths: List[Path]) -> np.ndarray:
        """Preprocess a list of images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Preprocessed images as numpy array
            
        Raises:
            DataError: If image loading fails
        """
        processed_images = []
        
        for img_path in image_paths:
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                
                # Apply transforms
                img_tensor = self.test_transform(img)
                processed_images.append(img_tensor.numpy())
                
            except Exception as e:
                raise DataError(f"Failed to preprocess image {img_path}: {str(e)}")
        
        return np.array(processed_images)
    
    def preprocess_labels(self, labels_df: pd.DataFrame) -> np.ndarray:
        """Preprocess labels for multi-label classification.
        
        Args:
            labels_df: DataFrame with label columns or 'Finding Labels' column
            
        Returns:
            Binary label matrix (n_samples, n_classes)
        """
        # NIH Chest X-ray pathology classes (14 classes + No Finding)
        pathology_classes = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia', 'No Finding'
        ]
        
        # Check if labels are in 'Finding Labels' column (NIH format)
        if 'Finding Labels' in labels_df.columns:
            logger.info("Processing NIH Chest X-ray labels from 'Finding Labels' column")
            
            # Create binary matrix
            labels_matrix = np.zeros((len(labels_df), len(pathology_classes)))
            
            for idx, row in labels_df.iterrows():
                findings = str(row['Finding Labels']).split('|')
                for finding in findings:
                    finding = finding.strip()
                    if finding in pathology_classes:
                        class_idx = pathology_classes.index(finding)
                        labels_matrix[idx, class_idx] = 1
            
            logger.info(f"Processed {len(pathology_classes)} pathology labels")
            return labels_matrix
        
        # Otherwise, check for individual columns (alternative format)
        else:
            available_columns = [col for col in pathology_classes if col in labels_df.columns]
            
            if not available_columns:
                raise DataError("No pathology columns found in labels DataFrame")
            
            logger.info(f"Using {len(available_columns)} pathology labels")
            
            # Convert to binary (1 for positive, 0 for negative/uncertain)
            labels = labels_df[available_columns].fillna(0)
            labels = (labels == 1.0).astype(int)
            
            return labels.values
    
    def create_train_val_test_split(
        self,
        dataset: Dataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Create stratified train/val/test split.
        
        Args:
            dataset: Complete dataset
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        # Set random seed
        torch.manual_seed(seed)
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        logger.info(f"Splitting dataset: train={train_size}, val={val_size}, test={test_size}")
        
        # Perform random split
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        return train_dataset, val_dataset, test_dataset


class MIMICCXRDataset(Dataset):
    """PyTorch Dataset for MIMIC-CXR.
    
    Loads images and labels on-the-fly to save memory.
    """
    
    def __init__(
        self,
        image_paths: List[Path],
        labels: np.ndarray,
        transform: transforms.Compose = None
    ):
        """Initialize dataset.
        
        Args:
            image_paths: List of paths to image files
            labels: Label array (n_samples, n_classes)
            transform: Optional transforms to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        assert len(image_paths) == len(labels), \
            "Number of images and labels must match"
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.image_paths)
    
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
