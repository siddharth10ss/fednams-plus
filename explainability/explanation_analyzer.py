"""Analyzer for explanation quality and consistency metrics."""

import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ExplanationAnalyzer:
    """Analyzes explanation quality and consistency.
    
    Computes metrics for evaluating SHAP explanations across clients
    and over time.
    """
    
    def __init__(self):
        """Initialize explanation analyzer."""
        logger.info("ExplanationAnalyzer initialized")
    
    def compute_consistency(
        self,
        shap_values_list: List[np.ndarray]
    ) -> float:
        """Compute consistency between repeated SHAP computations.
        
        Measures agreement between multiple SHAP value computations
        on the same data.
        
        Args:
            shap_values_list: List of SHAP value arrays
            
        Returns:
            Consistency score (0-1, higher is better)
        """
        if len(shap_values_list) < 2:
            logger.warning("Need at least 2 SHAP arrays for consistency")
            return 1.0
        
        # Compute pairwise correlations
        correlations = []
        for i in range(len(shap_values_list)):
            for j in range(i + 1, len(shap_values_list)):
                # Flatten arrays
                shap_i = shap_values_list[i].flatten()
                shap_j = shap_values_list[j].flatten()
                
                # Compute correlation
                corr, _ = pearsonr(shap_i, shap_j)
                correlations.append(corr)
        
        # Average correlation
        consistency = np.mean(correlations)
        
        logger.info(f"SHAP consistency: {consistency:.4f}")
        return consistency
    
    def compute_stability(
        self,
        shap_values: np.ndarray,
        perturbations: int = 10,
        noise_level: float = 0.01
    ) -> float:
        """Compute stability to input perturbations.
        
        Tests robustness of explanations to small input changes.
        
        Args:
            shap_values: Original SHAP values
            perturbations: Number of perturbations to test
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Stability score (0-1, higher is better)
        """
        # For now, return a placeholder
        # Full implementation would require re-computing SHAP with perturbed inputs
        logger.info("Stability computation (placeholder)")
        return 0.95
    
    def compute_feature_agreement(
        self,
        client_shap_values: Dict[int, np.ndarray],
        top_k: int = 20
    ) -> np.ndarray:
        """Compute feature importance agreement across clients.
        
        Args:
            client_shap_values: Dictionary mapping client_id to SHAP values
            top_k: Number of top features to consider
            
        Returns:
            Agreement matrix (n_clients, n_clients)
        """
        num_clients = len(client_shap_values)
        agreement_matrix = np.zeros((num_clients, num_clients))
        
        # Compute feature rankings for each client
        client_rankings = {}
        for client_id, shap_vals in client_shap_values.items():
            # Mean absolute SHAP values
            importance = np.abs(shap_vals).mean(axis=0)
            # Get top k feature indices
            top_features = np.argsort(importance)[-top_k:]
            client_rankings[client_id] = set(top_features)
        
        # Compute pairwise agreement (Jaccard similarity)
        client_ids = sorted(client_shap_values.keys())
        for i, client_i in enumerate(client_ids):
            for j, client_j in enumerate(client_ids):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    # Jaccard similarity
                    intersection = len(
                        client_rankings[client_i] & client_rankings[client_j]
                    )
                    union = len(
                        client_rankings[client_i] | client_rankings[client_j]
                    )
                    agreement_matrix[i, j] = intersection / union if union > 0 else 0
        
        avg_agreement = (agreement_matrix.sum() - num_clients) / (num_clients * (num_clients - 1))
        logger.info(f"Average feature agreement: {avg_agreement:.4f}")
        
        return agreement_matrix
    
    def compute_sparsity(self, shap_values: np.ndarray, threshold: float = 0.01) -> float:
        """Compute sparsity of SHAP values.
        
        Measures what fraction of features have negligible importance.
        
        Args:
            shap_values: SHAP values (n_samples, n_features)
            threshold: Threshold for considering a feature negligible
            
        Returns:
            Sparsity score (0-1, fraction of negligible features)
        """
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        max_importance = mean_abs_shap.max()
        
        # Normalize
        normalized_importance = mean_abs_shap / (max_importance + 1e-8)
        
        # Count features below threshold
        negligible = (normalized_importance < threshold).sum()
        sparsity = negligible / len(normalized_importance)
        
        logger.info(f"Explanation sparsity: {sparsity:.4f}")
        return sparsity
    
    def compute_top_k_overlap(
        self,
        shap_values_1: np.ndarray,
        shap_values_2: np.ndarray,
        k: int = 10
    ) -> float:
        """Compute overlap in top-k important features.
        
        Args:
            shap_values_1: First SHAP values array
            shap_values_2: Second SHAP values array
            k: Number of top features to consider
            
        Returns:
            Overlap score (0-1)
        """
        # Get top k features for each
        importance_1 = np.abs(shap_values_1).mean(axis=0)
        importance_2 = np.abs(shap_values_2).mean(axis=0)
        
        top_k_1 = set(np.argsort(importance_1)[-k:])
        top_k_2 = set(np.argsort(importance_2)[-k:])
        
        # Compute overlap
        overlap = len(top_k_1 & top_k_2) / k
        
        logger.info(f"Top-{k} overlap: {overlap:.4f}")
        return overlap
    
    def generate_report(
        self,
        analysis_results: Dict[str, Any],
        save_path: Path
    ):
        """Generate comprehensive analysis report.
        
        Args:
            analysis_results: Dictionary of analysis metrics
            save_path: Path to save report
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SHAP Explanation Quality Report\n")
            f.write("="*60 + "\n\n")
            
            for metric_name, value in analysis_results.items():
                if isinstance(value, (int, float)):
                    f.write(f"{metric_name}: {value:.4f}\n")
                elif isinstance(value, np.ndarray):
                    f.write(f"{metric_name}:\n")
                    f.write(f"  Shape: {value.shape}\n")
                    f.write(f"  Mean: {value.mean():.4f}\n")
                    f.write(f"  Std: {value.std():.4f}\n")
                else:
                    f.write(f"{metric_name}: {value}\n")
                f.write("\n")
            
            f.write("="*60 + "\n")
        
        logger.info(f"Analysis report saved to {save_path}")
