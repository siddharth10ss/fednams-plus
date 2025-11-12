"""Conformal prediction for uncertainty quantification."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Set, Tuple, Optional
from torch.utils.data import DataLoader

from utils.exceptions import ModelError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ConformalPredictor:
    """Provides uncertainty quantification using conformal prediction.
    
    Implements Adaptive Prediction Sets (APS) method for multi-class
    classification with statistical coverage guarantees.
    """
    
    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        device: str = 'cuda'
    ):
        """Initialize conformal predictor.
        
        Args:
            model: Trained neural network model
            alpha: Miscoverage rate (1-alpha is target coverage)
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
        self.alpha = alpha
        
        # Calibration scores
        self.calibration_scores = None
        self.q_hat = None
        
        logger.info(
            f"ConformalPredictor initialized: "
            f"alpha={alpha}, target coverage={1-alpha:.2%}"
        )
    
    def calibrate(self, cal_dataloader: DataLoader):
        """Calibrate conformal predictor using calibration set.
        
        Args:
            cal_dataloader: DataLoader for calibration data
        """
        self.model.eval()
        
        all_scores = []
        
        with torch.no_grad():
            for images, labels in cal_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get model predictions (logits)
                outputs = self.model(images)
                
                # Convert to probabilities
                probs = torch.sigmoid(outputs)
                
                # Compute conformity scores for each sample
                # For multi-label: use sum of probabilities for positive labels
                scores = self._compute_scores(probs, labels)
                all_scores.extend(scores.cpu().numpy())
        
        self.calibration_scores = np.array(all_scores)
        
        # Compute quantile
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(self.calibration_scores, q_level)
        
        logger.info(
            f"Calibration complete: "
            f"{len(self.calibration_scores)} samples, "
            f"q_hat={self.q_hat:.4f}"
        )
    
    def _compute_scores(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute conformity scores.
        
        For multi-label classification, we use the sum of probabilities
        for the true positive labels.
        
        Args:
            probs: Predicted probabilities (batch_size, num_classes)
            labels: True labels (batch_size, num_classes)
            
        Returns:
            Conformity scores (batch_size,)
        """
        # For each sample, sum probabilities of true positive labels
        # Higher score = better conformity
        scores = (probs * labels).sum(dim=1)
        
        # Invert so that lower score = better conformity (for quantile)
        scores = 1.0 - scores
        
        return scores
    
    def predict_with_sets(
        self,
        x: torch.Tensor,
        return_probs: bool = False
    ) -> Tuple[torch.Tensor, List[Set[int]]]:
        """Make predictions with prediction sets.
        
        Args:
            x: Input images (batch_size, 3, height, width)
            return_probs: Whether to return probabilities
            
        Returns:
            Tuple of:
                - predictions: Point predictions (batch_size, num_classes)
                - prediction_sets: List of prediction sets (one per sample)
        """
        if self.q_hat is None:
            raise ModelError("Conformal predictor not calibrated. Call calibrate() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.model(x)
            probs = torch.sigmoid(outputs)
        
        # Point predictions (threshold at 0.5)
        predictions = (probs > 0.5).float()
        
        # Construct prediction sets
        prediction_sets = []
        
        for i in range(len(x)):
            prob_sample = probs[i].cpu().numpy()
            
            # Sort classes by probability (descending)
            sorted_indices = np.argsort(-prob_sample)
            
            # Add classes until cumulative probability exceeds threshold
            pred_set = set()
            cumulative_prob = 0.0
            
            for idx in sorted_indices:
                pred_set.add(int(idx))
                cumulative_prob += prob_sample[idx]
                
                # Check if we've included enough classes
                if (1.0 - cumulative_prob) <= self.q_hat:
                    break
            
            prediction_sets.append(pred_set)
        
        if return_probs:
            return predictions, prediction_sets, probs
        else:
            return predictions, prediction_sets
    
    def compute_coverage(
        self,
        dataloader: DataLoader
    ) -> Tuple[float, float]:
        """Compute empirical coverage on test data.
        
        Args:
            dataloader: Test data loader
            
        Returns:
            Tuple of (coverage, average_set_size)
        """
        if self.q_hat is None:
            raise ModelError("Conformal predictor not calibrated.")
        
        self.model.eval()
        
        total_samples = 0
        covered_samples = 0
        total_set_size = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.cpu().numpy()
                
                # Get prediction sets
                _, prediction_sets = self.predict_with_sets(images)
                
                # Check coverage for each sample
                for i, pred_set in enumerate(prediction_sets):
                    # Get true positive labels
                    true_labels = set(np.where(labels[i] == 1)[0])
                    
                    # Check if all true labels are in prediction set
                    if true_labels.issubset(pred_set):
                        covered_samples += 1
                    
                    total_set_size += len(pred_set)
                    total_samples += 1
        
        coverage = covered_samples / total_samples
        avg_set_size = total_set_size / total_samples
        
        logger.info(
            f"Coverage: {coverage:.4f} (target: {1-self.alpha:.4f}), "
            f"Avg set size: {avg_set_size:.2f}"
        )
        
        return coverage, avg_set_size
    
    def compute_conditional_coverage(
        self,
        dataloader: DataLoader,
        num_classes: int
    ) -> np.ndarray:
        """Compute coverage per class.
        
        Args:
            dataloader: Test data loader
            num_classes: Number of classes
            
        Returns:
            Per-class coverage (num_classes,)
        """
        if self.q_hat is None:
            raise ModelError("Conformal predictor not calibrated.")
        
        self.model.eval()
        
        class_counts = np.zeros(num_classes)
        class_covered = np.zeros(num_classes)
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.cpu().numpy()
                
                # Get prediction sets
                _, prediction_sets = self.predict_with_sets(images)
                
                # Check coverage for each class
                for i, pred_set in enumerate(prediction_sets):
                    for class_idx in range(num_classes):
                        if labels[i, class_idx] == 1:
                            class_counts[class_idx] += 1
                            if class_idx in pred_set:
                                class_covered[class_idx] += 1
        
        # Compute per-class coverage
        conditional_coverage = np.divide(
            class_covered,
            class_counts,
            out=np.zeros_like(class_covered),
            where=class_counts > 0
        )
        
        logger.info(f"Conditional coverage computed for {num_classes} classes")
        
        return conditional_coverage
    
    def get_prediction_intervals(
        self,
        dataloader: DataLoader
    ) -> List[Tuple[float, float]]:
        """Get prediction interval widths as uncertainty measure.
        
        Args:
            dataloader: Data loader
            
        Returns:
            List of (set_size, entropy) tuples for each sample
        """
        if self.q_hat is None:
            raise ModelError("Conformal predictor not calibrated.")
        
        self.model.eval()
        
        intervals = []
        
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                
                # Get predictions with sets and probabilities
                _, prediction_sets, probs = self.predict_with_sets(
                    images, return_probs=True
                )
                
                # Compute uncertainty measures
                for i, pred_set in enumerate(prediction_sets):
                    set_size = len(pred_set)
                    
                    # Compute entropy as additional uncertainty measure
                    prob_sample = probs[i].cpu().numpy()
                    entropy = -np.sum(
                        prob_sample * np.log(prob_sample + 1e-10) +
                        (1 - prob_sample) * np.log(1 - prob_sample + 1e-10)
                    )
                    
                    intervals.append((set_size, entropy))
        
        return intervals
