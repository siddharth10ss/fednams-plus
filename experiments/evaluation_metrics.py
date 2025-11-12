"""Evaluation metrics for model performance assessment."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Set, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class EvaluationMetrics:
    """Computes comprehensive evaluation metrics.
    
    Provides static methods for computing classification, explanation,
    uncertainty, and communication metrics.
    """
    
    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Compute classification metrics.
        
        Args:
            y_true: True labels (n_samples, n_classes)
            y_pred: Predicted labels (n_samples, n_classes)
            y_prob: Predicted probabilities (n_samples, n_classes)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(
            y_true.flatten(),
            y_pred.flatten()
        )
        
        # Precision, Recall, F1 (macro and micro)
        for avg in ['macro', 'micro']:
            metrics[f'precision_{avg}'] = precision_score(
                y_true, y_pred, average=avg, zero_division=0
            )
            metrics[f'recall_{avg}'] = recall_score(
                y_true, y_pred, average=avg, zero_division=0
            )
            metrics[f'f1_{avg}'] = f1_score(
                y_true, y_pred, average=avg, zero_division=0
            )
        
        # AUC-ROC and AUC-PR
        try:
            metrics['auc_roc_macro'] = roc_auc_score(
                y_true, y_prob, average='macro'
            )
            metrics['auc_roc_micro'] = roc_auc_score(
                y_true, y_prob, average='micro'
            )
        except ValueError as e:
            logger.warning(f"Could not compute AUC-ROC: {str(e)}")
            metrics['auc_roc_macro'] = 0.0
            metrics['auc_roc_micro'] = 0.0
        
        try:
            metrics['auc_pr_macro'] = average_precision_score(
                y_true, y_prob, average='macro'
            )
            metrics['auc_pr_micro'] = average_precision_score(
                y_true, y_prob, average='micro'
            )
        except ValueError as e:
            logger.warning(f"Could not compute AUC-PR: {str(e)}")
            metrics['auc_pr_macro'] = 0.0
            metrics['auc_pr_micro'] = 0.0
        
        logger.info(
            f"Classification metrics: "
            f"Acc={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1_macro']:.4f}, "
            f"AUC={metrics['auc_roc_macro']:.4f}"
        )
        
        return metrics
    
    @staticmethod
    def compute_explanation_metrics(
        shap_values: np.ndarray
    ) -> Dict[str, float]:
        """Compute explanation quality metrics.
        
        Args:
            shap_values: SHAP values (n_samples, n_features)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Mean absolute SHAP value
        metrics['mean_abs_shap'] = np.abs(shap_values).mean()
        
        # SHAP variance
        metrics['shap_variance'] = np.var(shap_values)
        
        # Sparsity (fraction of near-zero SHAP values)
        threshold = 0.01 * np.abs(shap_values).max()
        metrics['sparsity'] = (np.abs(shap_values) < threshold).mean()
        
        # Top-k concentration (what fraction of importance is in top 10%)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_importance = np.sort(mean_abs_shap)[::-1]
        k = max(1, len(sorted_importance) // 10)
        metrics['top_k_concentration'] = sorted_importance[:k].sum() / sorted_importance.sum()
        
        logger.info(
            f"Explanation metrics: "
            f"Mean|SHAP|={metrics['mean_abs_shap']:.4f}, "
            f"Sparsity={metrics['sparsity']:.4f}"
        )
        
        return metrics
    
    @staticmethod
    def compute_uncertainty_metrics(
        predictions: List[Set[int]],
        y_true: np.ndarray,
        alpha: float
    ) -> Dict[str, float]:
        """Compute uncertainty quantification metrics.
        
        Args:
            predictions: List of prediction sets
            y_true: True labels (n_samples, n_classes)
            alpha: Target miscoverage rate
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Coverage
        covered = 0
        for i, pred_set in enumerate(predictions):
            true_labels = set(np.where(y_true[i] == 1)[0])
            if true_labels.issubset(pred_set):
                covered += 1
        
        metrics['coverage'] = covered / len(predictions)
        metrics['target_coverage'] = 1 - alpha
        metrics['coverage_gap'] = abs(metrics['coverage'] - metrics['target_coverage'])
        
        # Average set size
        metrics['avg_set_size'] = np.mean([len(s) for s in predictions])
        
        # Set size variance
        metrics['set_size_variance'] = np.var([len(s) for s in predictions])
        
        logger.info(
            f"Uncertainty metrics: "
            f"Coverage={metrics['coverage']:.4f} "
            f"(target={metrics['target_coverage']:.4f}), "
            f"Avg set size={metrics['avg_set_size']:.2f}"
        )
        
        return metrics
    
    @staticmethod
    def compute_communication_cost(
        model: nn.Module,
        num_rounds: int,
        num_clients: int
    ) -> Dict[str, float]:
        """Compute communication cost metrics.
        
        Args:
            model: Neural network model
            num_rounds: Number of federated rounds
            num_clients: Number of clients
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        metrics['num_parameters'] = num_params
        
        # Total parameters transmitted (up + down for each client each round)
        metrics['total_params_transmitted'] = num_params * num_clients * num_rounds * 2
        
        # MB transmitted (assuming float32)
        metrics['mb_transmitted'] = metrics['total_params_transmitted'] * 4 / (1024**2)
        
        # MB per round
        metrics['mb_per_round'] = metrics['mb_transmitted'] / num_rounds
        
        # Compression ratio (if applicable)
        metrics['compression_ratio'] = 1.0  # No compression by default
        
        logger.info(
            f"Communication cost: "
            f"{metrics['mb_transmitted']:.2f} MB total, "
            f"{metrics['mb_per_round']:.2f} MB/round"
        )
        
        return metrics
    
    @staticmethod
    def compute_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Compute per-class metrics.
        
        Args:
            y_true: True labels (n_samples, n_classes)
            y_pred: Predicted labels (n_samples, n_classes)
            y_prob: Predicted probabilities (n_samples, n_classes)
            class_names: Optional class names
            
        Returns:
            Dictionary of per-class metrics
        """
        num_classes = y_true.shape[1]
        
        metrics = {
            'precision': np.zeros(num_classes),
            'recall': np.zeros(num_classes),
            'f1': np.zeros(num_classes),
            'support': np.zeros(num_classes)
        }
        
        for i in range(num_classes):
            # Support (number of positive samples)
            metrics['support'][i] = y_true[:, i].sum()
            
            if metrics['support'][i] > 0:
                metrics['precision'][i] = precision_score(
                    y_true[:, i], y_pred[:, i], zero_division=0
                )
                metrics['recall'][i] = recall_score(
                    y_true[:, i], y_pred[:, i], zero_division=0
                )
                metrics['f1'][i] = f1_score(
                    y_true[:, i], y_pred[:, i], zero_division=0
                )
        
        logger.info(f"Per-class metrics computed for {num_classes} classes")
        
        return metrics
    
    @staticmethod
    def compute_fairness_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        client_ids: np.ndarray
    ) -> Dict[str, float]:
        """Compute fairness metrics across clients.
        
        Args:
            y_true: True labels (n_samples, n_classes)
            y_pred: Predicted labels (n_samples, n_classes)
            client_ids: Client ID for each sample (n_samples,)
            
        Returns:
            Dictionary of fairness metrics
        """
        metrics = {}
        
        # Compute accuracy per client
        unique_clients = np.unique(client_ids)
        client_accuracies = []
        
        for client_id in unique_clients:
            mask = client_ids == client_id
            if mask.sum() > 0:
                acc = accuracy_score(
                    y_true[mask].flatten(),
                    y_pred[mask].flatten()
                )
                client_accuracies.append(acc)
        
        # Fairness metrics
        metrics['min_accuracy'] = np.min(client_accuracies)
        metrics['max_accuracy'] = np.max(client_accuracies)
        metrics['accuracy_std'] = np.std(client_accuracies)
        metrics['accuracy_range'] = metrics['max_accuracy'] - metrics['min_accuracy']
        
        logger.info(
            f"Fairness metrics: "
            f"Acc range=[{metrics['min_accuracy']:.4f}, {metrics['max_accuracy']:.4f}], "
            f"std={metrics['accuracy_std']:.4f}"
        )
        
        return metrics
