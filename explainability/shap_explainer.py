"""SHAP explainer for computing feature importance in NAM models."""

import torch
import torch.nn as nn
import numpy as np
import shap
from typing import Tuple, Optional
from torch.utils.data import DataLoader

from utils.exceptions import ExplanationError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class SHAPExplainer:
    """Computes SHAP values for model predictions.
    
    Uses DeepSHAP for neural network explanations at the feature level.
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: torch.Tensor,
        device: str = 'cuda'
    ):
        """Initialize SHAP explainer.
        
        Args:
            model: Neural network model to explain
            background_data: Background samples for SHAP (n_samples, feature_dim)
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
        self.background_data = background_data.to(device)
        
        # Create DeepSHAP explainer
        try:
            self.explainer = shap.DeepExplainer(
                self.model,
                self.background_data
            )
            logger.info(
                f"SHAPExplainer initialized with "
                f"{background_data.shape[0]} background samples"
            )
        except Exception as e:
            raise ExplanationError(f"Failed to initialize SHAP explainer: {str(e)}")
    
    def explain(
        self,
        data: torch.Tensor,
        num_samples: Optional[int] = None
    ) -> np.ndarray:
        """Compute SHAP values for given data.
        
        Args:
            data: Input data (n_samples, ...)
            num_samples: Optional limit on number of samples to explain
            
        Returns:
            SHAP values (n_samples, feature_dim, num_classes)
        """
        self.model.eval()
        
        # Limit samples if specified
        if num_samples is not None and len(data) > num_samples:
            indices = np.random.choice(len(data), num_samples, replace=False)
            data = data[indices]
        
        data = data.to(self.device)
        
        try:
            with torch.no_grad():
                shap_values = self.explainer.shap_values(data)
            
            # Convert to numpy
            if isinstance(shap_values, list):
                # Multi-class case
                shap_values = np.array(shap_values)
                # Transpose to (n_samples, feature_dim, num_classes)
                shap_values = np.transpose(shap_values, (1, 2, 0))
            else:
                shap_values = np.array(shap_values)
            
            logger.info(f"Computed SHAP values: {shap_values.shape}")
            return shap_values
            
        except Exception as e:
            raise ExplanationError(f"SHAP computation failed: {str(e)}")
    
    def explain_batch(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SHAP values for batched data.
        
        Args:
            dataloader: Data loader
            max_samples: Maximum number of samples to process
            
        Returns:
            Tuple of (shap_values, base_values)
        """
        self.model.eval()
        
        all_shap_values = []
        all_data = []
        total_samples = 0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            if max_samples is not None and total_samples >= max_samples:
                break
            
            images = images.to(self.device)
            
            # Compute SHAP for this batch
            try:
                with torch.no_grad():
                    shap_values = self.explainer.shap_values(images)
                
                if isinstance(shap_values, list):
                    shap_values = np.array(shap_values)
                    shap_values = np.transpose(shap_values, (1, 2, 0))
                else:
                    shap_values = np.array(shap_values)
                
                all_shap_values.append(shap_values)
                all_data.append(images.cpu().numpy())
                
                total_samples += len(images)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {total_samples} samples...")
                    
            except Exception as e:
                logger.warning(f"Failed to compute SHAP for batch {batch_idx}: {str(e)}")
                continue
        
        # Concatenate all batches
        all_shap_values = np.concatenate(all_shap_values, axis=0)
        all_data = np.concatenate(all_data, axis=0)
        
        # Compute base values (expected value)
        with torch.no_grad():
            base_values = self.model(self.background_data).cpu().numpy().mean(axis=0)
        
        logger.info(
            f"Batch SHAP computation complete: "
            f"{all_shap_values.shape[0]} samples"
        )
        
        return all_shap_values, base_values
    
    @staticmethod
    def select_background_samples(
        data: torch.Tensor,
        num_samples: int = 100,
        method: str = 'kmeans'
    ) -> torch.Tensor:
        """Select representative background samples.
        
        Args:
            data: Full dataset (n_samples, ...)
            num_samples: Number of background samples to select
            method: Selection method ('kmeans', 'random')
            
        Returns:
            Selected background samples
        """
        if method == 'random':
            indices = np.random.choice(len(data), num_samples, replace=False)
            return data[indices]
        
        elif method == 'kmeans':
            try:
                from sklearn.cluster import KMeans
                
                # Flatten data for clustering
                data_flat = data.reshape(len(data), -1).cpu().numpy()
                
                # Perform k-means
                kmeans = KMeans(n_clusters=num_samples, random_state=42)
                kmeans.fit(data_flat)
                
                # Find closest sample to each centroid
                distances = kmeans.transform(data_flat)
                closest_indices = distances.argmin(axis=0)
                
                return data[closest_indices]
                
            except ImportError:
                logger.warning("scikit-learn not available, using random sampling")
                return SHAPExplainer.select_background_samples(
                    data, num_samples, method='random'
                )
        
        else:
            raise ValueError(f"Unknown selection method: {method}")
