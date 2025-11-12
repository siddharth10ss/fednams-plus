"""SHAP visualization for generating explanation plots."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import shap

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class SHAPVisualizer:
    """Generates visualizations for SHAP explanations.
    
    Creates publication-ready plots for feature importance analysis.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """Initialize SHAP visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        sns.set_palette("husl")
        logger.info("SHAPVisualizer initialized")
    
    def plot_summary(
        self,
        shap_values: np.ndarray,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        max_display: int = 20
    ):
        """Generate SHAP summary plot.
        
        Args:
            shap_values: SHAP values (n_samples, n_features)
            feature_names: Names of features
            save_path: Path to save plot
            max_display: Maximum number of features to display
        """
        plt.figure(figsize=(10, 8))
        
        try:
            shap.summary_plot(
                shap_values,
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
            
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Summary plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Failed to create summary plot: {str(e)}")
        finally:
            plt.close()
    
    def plot_feature_importance(
        self,
        shap_values: np.ndarray,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        top_k: int = 20
    ):
        """Generate feature importance bar plot.
        
        Args:
            shap_values: SHAP values (n_samples, n_features)
            feature_names: Names of features
            save_path: Path to save plot
            top_k: Number of top features to show
        """
        # Compute mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Get top k features
        top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
        top_values = mean_abs_shap[top_indices]
        
        if feature_names is not None:
            top_names = [feature_names[i] for i in top_indices]
        else:
            top_names = [f"Feature {i}" for i in top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(f'Top {top_k} Feature Importances')
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_dependence(
        self,
        shap_values: np.ndarray,
        feature_idx: int,
        feature_values: np.ndarray,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None
    ):
        """Generate SHAP dependence plot for a specific feature.
        
        Args:
            shap_values: SHAP values (n_samples, n_features)
            feature_idx: Index of feature to plot
            feature_values: Actual feature values (n_samples, n_features)
            feature_names: Names of features
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        
        feature_name = (
            feature_names[feature_idx] if feature_names
            else f"Feature {feature_idx}"
        )
        
        try:
            shap.dependence_plot(
                feature_idx,
                shap_values,
                feature_values,
                feature_names=feature_names,
                show=False
            )
            
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Dependence plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Failed to create dependence plot: {str(e)}")
        finally:
            plt.close()
    
    def plot_client_comparison(
        self,
        client_shap_values: Dict[int, np.ndarray],
        save_path: Optional[Path] = None,
        top_k: int = 15
    ):
        """Compare feature importances across clients.
        
        Args:
            client_shap_values: Dictionary mapping client_id to SHAP values
            save_path: Path to save plot
            top_k: Number of top features to show
        """
        # Compute mean absolute SHAP for each client
        client_importances = {}
        for client_id, shap_vals in client_shap_values.items():
            client_importances[client_id] = np.abs(shap_vals).mean(axis=0)
        
        # Get global top features
        global_importance = np.mean(
            list(client_importances.values()), axis=0
        )
        top_indices = np.argsort(global_importance)[-top_k:][::-1]
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(top_k)
        width = 0.8 / len(client_importances)
        
        for i, (client_id, importances) in enumerate(client_importances.items()):
            offset = (i - len(client_importances)/2) * width
            ax.bar(
                x + offset,
                importances[top_indices],
                width,
                label=f'Client {client_id}'
            )
        
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Mean |SHAP value|')
        ax.set_title(f'Feature Importance Comparison Across Clients (Top {top_k})')
        ax.set_xticks(x)
        ax.set_xticklabels([f'F{i}' for i in top_indices])
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Client comparison plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_heatmap(
        self,
        shap_values: np.ndarray,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        top_k: int = 20
    ):
        """Generate heatmap of SHAP values.
        
        Args:
            shap_values: SHAP values (n_samples, n_features)
            feature_names: Names of features
            save_path: Path to save plot
            top_k: Number of top features to show
        """
        # Get top k features by importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
        
        # Select top features
        shap_subset = shap_values[:, top_indices]
        
        if feature_names is not None:
            top_names = [feature_names[i] for i in top_indices]
        else:
            top_names = [f"F{i}" for i in top_indices]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(
            shap_subset.T,
            cmap='RdBu_r',
            center=0,
            yticklabels=top_names,
            cbar_kws={'label': 'SHAP value'},
            ax=ax
        )
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Feature')
        ax.set_title(f'SHAP Values Heatmap (Top {top_k} Features)')
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
