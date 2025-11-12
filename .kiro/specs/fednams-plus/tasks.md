# Implementation Plan

- [x] 1. Project setup and core infrastructure



  - Create project directory structure with modules: data/, models/, training/, explainability/, experiments/, tests/, configs/, scripts/
  - Create requirements.txt with all dependencies: torch, torchvision, numpy, pandas, scikit-learn, shap, matplotlib, seaborn, pyyaml, tqdm
  - Implement configuration dataclasses (PreprocessConfig, ModelConfig, TrainingConfig, FedConfig, ExperimentConfig) in configs/config.py
  - Create custom exception hierarchy (FedNAMsError, DataError, ModelError, TrainingError, ExplanationError, ConfigurationError) in utils/exceptions.py
  - Set up logging utilities with file and console handlers in utils/logging.py
  - _Requirements: 7.1, 7.5_






- [-] 2. Data module implementation



- [x] 2.1 Implement data downloading and validation

  - Create DataDownloader class with download() and verify_integrity() methods supporting Kaggle API


  - Implement authentication handling and checksum verification
  - Add retry logic for network failures
  - _Requirements: 1.1, 1.5_




- [ ] 2.2 Implement data preprocessing pipeline
  - Create DataPreprocessor class with image resizing to 224x224


  - Implement ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  - Add data augmentation (random horizontal flip, rotation, brightness/contrast adjustment)
  - Implement multi-label encoding for MIMIC-CXR pathologies
  - Create stratified train/val/test splitting (70/15/15) with create_train_val_test_split()
  - _Requirements: 1.1, 11.3, 11.4_




- [ ] 2.3 Implement federated data partitioning
  - Create FederatedDataPartitioner class with partition() method


  - Implement Dirichlet distribution-based non-IID partitioning with configurable alpha parameter
  - Implement pathology-based partitioning strategy
  - Implement quantity-based partitioning with varying client dataset sizes
  - Add minimum viable data validation (ensure each client has at least 100 samples)
  - Create generate_statistics() method to compute label distributions, sample counts, and heterogeneity metrics
  - Save partition metadata to JSON files in client-specific directories
  - _Requirements: 1.2, 1.3, 1.4_




- [ ] 2.4 Create abstract dataset interface for extensibility
  - Define BaseDataset abstract class with load(), preprocess(), and get_labels() methods
  - Implement MIMICCXRDataset as concrete implementation
  - Add dataset registry pattern for easy dataset switching
  - Document steps for adding new datasets in data/README.md
  - _Requirements: 11.1, 11.2, 11.5_

- [x] 3. Models module implementation



- [x] 3.1 Implement feature extractor component


  - Create FeatureExtractor class inheriting from nn.Module
  - Support multiple CNN backbones (ResNet18, ResNet50, DenseNet121) using torchvision.models
  - Load pretrained ImageNet weights
  - Extract features from penultimate layer (512-d for ResNet18)
  - Implement freeze_backbone() and unfreeze_backbone() methods
  - _Requirements: 2.1_

- [x] 3.2 Implement NAM head architecture


  - Create NAMHead class with separate sub-networks for each feature dimension
  - Implement feature-wise MLPs with configurable hidden units [64, 32]
  - Add feature dropout (p=0.3) for regularization
  - Implement forward() returning both predictions and feature contributions
  - Create get_feature_contributions() method for interpretability
  - Optionally implement ExU (Exponential Units) activation for monotonicity
  - _Requirements: 2.2, 2.3, 2.5_

- [x] 3.3 Implement complete FedNAMsModel


  - Create FedNAMsModel class combining FeatureExtractor and NAMHead
  - Implement forward() for standard predictions
  - Implement forward_with_contributions() for interpretability analysis
  - Create get_parameters() and set_parameters() for federated communication
  - Support multi-class classification with 14 output classes for MIMIC-CXR
  - _Requirements: 2.3, 2.4, 3.6_

- [x] 3.4 Implement baseline models for comparison


  - Create FedAvgCNN class using standard ResNet/DenseNet architecture
  - Implement FedAvgCNN_GradCAM with Grad-CAM explanation capability
  - Create CentralizedNAM for privacy baseline comparison
  - Implement get_explanations() method for baseline models
  - _Requirements: 6.1_

- [ ]* 3.5 Write model unit tests
  - Test feature extractor output shapes with different backbones
  - Test NAM head forward pass and contribution extraction
  - Test parameter serialization and deserialization
  - Test model compatibility across different configurations
  - _Requirements: 12.1_

- [x] 4. Training module implementation



- [x] 4.1 Implement local trainer for client-side training


  - Create LocalTrainer class with train_epoch() and evaluate() methods
  - Support multiple optimizers (Adam, SGD, AdamW) with configurable learning rates
  - Implement learning rate schedulers (StepLR, CosineAnnealingLR)
  - Add mixed precision training using torch.cuda.amp
  - Implement early stopping with configurable patience (default: 10 epochs)
  - Track per-epoch metrics (loss, accuracy, AUC-ROC) and log to CSV
  - Implement gradient clipping to prevent exploding gradients
  - _Requirements: 3.1, 3.5, 10.1_

- [x] 4.2 Implement FedAvg aggregation


  - Create FedAvgAggregator class with aggregate() method
  - Implement weighted averaging based on client dataset sizes
  - Support uniform aggregation as alternative
  - Add parameter shape validation before aggregation
  - Implement weighted_average() helper for tensor operations
  - _Requirements: 3.2_

- [x] 4.3 Implement federated orchestrator


  - Create FederatedOrchestrator class coordinating training across clients
  - Implement run_round() to execute one federated round (local training + aggregation)
  - Implement run_training() for complete multi-round training
  - Add client sampling with configurable fraction (support training subset of clients per round)
  - Implement save_checkpoint() and load_checkpoint() for fault tolerance
  - Track global and per-client metrics across rounds
  - Log communication costs (count parameters transmitted per round)
  - Implement convergence detection for early termination
  - Distribute global model parameters to all clients after aggregation
  - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 10.5_

- [x] 4.4 Add progress monitoring and logging

  - Implement progress bars using tqdm for training rounds and epochs
  - Add estimated time remaining calculations
  - Create TensorBoard logging for training curves
  - Implement structured logging with timestamps and client IDs
  - _Requirements: 10.4, 13.2, 13.5_

- [ ]* 4.5 Write training module unit tests
  - Test local trainer epoch execution with synthetic data
  - Test FedAvg aggregation correctness with known parameters
  - Test weighted vs uniform aggregation
  - Test checkpoint saving and loading
  - Test early stopping logic
  - _Requirements: 12.1_

- [-] 5. Explainability module implementation

- [x] 5.1 Implement SHAP explainer


  - Create SHAPExplainer class using shap.DeepExplainer
  - Implement background data selection using k-means clustering (100-200 samples)
  - Create explain() method for single sample SHAP computation
  - Implement explain_batch() for efficient batch processing
  - Compute SHAP values at feature level (post-CNN extraction)
  - Return both SHAP values and base values
  - Add memory-efficient batch processing to handle large test sets
  - _Requirements: 4.1, 4.5, 10.2_

- [x] 5.2 Implement SHAP visualization



  - Create SHAPVisualizer class for generating plots
  - Implement plot_summary() for SHAP summary plots (beeswarm style)
  - Implement plot_feature_importance() for bar charts of mean absolute SHAP values
  - Implement plot_dependence() for feature dependence plots
  - Create plot_client_comparison() to compare feature importances across clients
  - Export plots in high-resolution PNG (300 DPI) and PDF formats
  - Store visualizations in client-specific output directories
  - _Requirements: 4.2, 4.3, 8.3, 8.4_



- [x] 5.3 Implement explanation quality analysis

  - Create ExplanationAnalyzer class for computing explanation metrics
  - Implement compute_consistency() measuring agreement between repeated SHAP computations
  - Implement compute_stability() testing robustness to input perturbations
  - Implement compute_feature_agreement() computing correlation across clients
  - Create generate_report() producing comprehensive analysis with statistical tests
  - Compute SHAP value statistics (mean absolute values, variance) across samples
  - _Requirements: 4.4, 4.5, 6.3_

- [ ]* 5.4 Write explainability module unit tests
  - Test SHAP value computation with simple models
  - Test explanation consistency metrics
  - Test visualization generation without errors
  - Test feature importance ranking
  - _Requirements: 12.1_

- [x] 6. Conformal prediction module implementation



- [x] 6.1 Implement conformal predictor


  - Create ConformalPredictor class with calibrate() and predict_with_sets() methods
  - Implement Adaptive Prediction Sets (APS) method for multi-class classification
  - Create calibration using held-out calibration set (15% of training data)
  - Implement score normalization to handle class imbalance
  - Support both client-specific and global calibration modes
  - Generate prediction sets with user-specified confidence levels (default: 0.9)
  - _Requirements: 5.1, 5.2_

- [x] 6.2 Implement uncertainty metrics computation

  - Implement compute_coverage() to measure empirical coverage vs target confidence
  - Compute average prediction set sizes
  - Calculate conditional coverage per class
  - Track per-client coverage statistics
  - Provide prediction interval widths as uncertainty measure
  - _Requirements: 5.3, 5.4, 5.5_

- [ ]* 6.3 Write conformal prediction unit tests
  - Test calibration procedure with synthetic data
  - Test prediction set generation
  - Test coverage computation accuracy
  - Test set size statistics
  - _Requirements: 12.1_

- [x] 7. Experiments module implementation



- [x] 7.1 Implement evaluation metrics computation


  - Create EvaluationMetrics class with static methods
  - Implement compute_classification_metrics() for accuracy, precision, recall, F1-score, AUC-ROC, AUC-PR
  - Implement compute_explanation_metrics() for SHAP consistency, feature stability, top-k agreement
  - Implement compute_uncertainty_metrics() for coverage, average set size, conditional coverage
  - Implement compute_communication_cost() tracking parameters transmitted and MB per round
  - _Requirements: 6.2, 6.4_

- [x] 7.2 Implement experiment runner


  - Create ExperimentRunner class with setup_experiment(), run_experiment(), and save_results() methods
  - Set random seeds for reproducibility (Python random, NumPy, PyTorch, CUDA)
  - Implement complete workflow: data loading → model creation → training → evaluation → SHAP → conformal prediction
  - Support multiple experiment types (FedNAMs+, baselines, ablations)
  - Implement evaluate_model() for comprehensive model evaluation on test set
  - Generate all required outputs (checkpoints, logs, metrics, visualizations)
  - Support experiment resumption from checkpoints
  - _Requirements: 8.1, 8.2, 13.3_

- [x] 7.3 Implement result visualization and export

  - Create visualization functions for training curves (loss and accuracy over rounds)
  - Generate comparison tables for all models (classification, explanation, uncertainty metrics)
  - Export metrics to CSV and JSON formats
  - Create publication-ready comparison charts using matplotlib/seaborn
  - Generate comprehensive results summary report
  - _Requirements: 6.5, 8.3, 8.4_

- [ ]* 7.4 Write integration tests
  - Test end-to-end training workflow with minimal synthetic dataset
  - Verify model convergence
  - Verify checkpoint creation and loading
  - Test SHAP computation after training
  - Test complete experiment execution
  - _Requirements: 12.2_

- [x] 8. Configuration and CLI implementation


- [x] 8.1 Create YAML configuration system


  - Define experiment configuration schema in configs/experiment.yaml
  - Implement configuration loading and validation
  - Support configuration overrides via command-line arguments
  - Create example configurations for different experiment types
  - _Requirements: 7.3_

- [x] 8.2 Implement command-line interface

  - Create CLI using argparse for experiment launching
  - Support commands: run, resume, evaluate, list-experiments
  - Implement parameter passing for all configurable options
  - Add help documentation for all commands and parameters
  - Display real-time training progress in terminal
  - _Requirements: 13.1, 13.2, 13.3_

- [x] 9. Privacy and security implementation



- [x] 9.1 Implement privacy safeguards

  - Validate that only model parameters are transmitted (no raw data)
  - Implement audit logging for all data access and model transmission events
  - Add optional differential privacy using Opacus library (gradient noise injection)
  - Implement privacy budget tracking (epsilon, delta) for DP-SGD
  - _Requirements: 3.6, 9.1, 9.3, 9.4_

- [x] 9.2 Create privacy documentation




  - Document privacy guarantees and data flow
  - Create privacy impact assessment document
  - Document SHAP explanation privacy properties based on literature
  - Provide GDPR/HIPAA compliance checklist
  - Document potential privacy risks and mitigations




  - _Requirements: 9.2, 9.5_

- [ ] 10. Documentation and reproducibility
- [ ] 10.1 Create comprehensive README
  - Write project overview and motivation
  - Document installation instructions for local and Colab environments
  - Provide usage examples with code snippets
  - Document all configuration options
  - Add troubleshooting section
  - Include citation information
  - _Requirements: 8.6_

- [x] 10.2 Create Jupyter notebooks for Colab



  - Create end-to-end training notebook (notebooks/train_fednams.ipynb)
  - Create evaluation and visualization notebook (notebooks/evaluate_results.ipynb)
  - Create baseline comparison notebook (notebooks/compare_baselines.ipynb)
  - Add Google Drive mounting and data loading instructions
  - Include inline documentation and markdown explanations
  - _Requirements: 8.5_

- [ ] 10.3 Add module-level documentation
  - Write comprehensive docstrings for all public classes and functions
  - Add type hints to all function signatures
  - Create module-level README files explaining purpose and usage
  - Document design decisions and architectural choices
  - _Requirements: 7.4_

- [ ]* 10.4 Set up continuous integration
  - Create GitHub Actions workflow (.github/workflows/ci.yml)
  - Configure testing on Python 3.8, 3.9, 3.10
  - Add linting with flake8 and formatting with black
  - Add type checking with mypy
  - Generate and upload coverage reports
  - Enforce 70% minimum coverage
  - _Requirements: 12.5_

- [ ] 11. Baseline experiments and evaluation
- [ ] 11.1 Implement baseline experiment scripts
  - Create scripts/run_fednams_plus.py for main FedNAMs+ experiment
  - Create scripts/run_fedavg_cnn.py for FedAvg CNN baseline
  - Create scripts/run_fedavg_gradcam.py for FedAvg with Grad-CAM
  - Create scripts/run_centralized_nam.py for centralized NAM baseline
  - _Requirements: 6.1_

- [ ] 11.2 Run comprehensive evaluation
  - Execute all baseline experiments with consistent hyperparameters
  - Collect classification metrics (accuracy, AUC, F1) for all models
  - Collect explanation quality metrics (SHAP consistency, stability)
  - Collect uncertainty metrics (coverage, set size)
  - Measure communication costs for federated methods
  - _Requirements: 6.2, 6.3, 6.4_

- [ ] 11.3 Generate comparison visualizations
  - Create comparison tables with all metrics across models
  - Generate bar charts comparing classification performance
  - Create plots comparing explanation quality
  - Visualize uncertainty calibration across methods
  - Generate communication cost comparison charts
  - _Requirements: 6.5_

- [ ] 12. Performance optimization
- [ ] 12.1 Implement feature caching
  - Cache extracted CNN features to disk after first epoch
  - Implement feature loading for subsequent epochs
  - Add cache invalidation on model changes
  - _Requirements: 10.1_

- [ ] 12.2 Add memory optimization
  - Implement gradient accumulation for large batch sizes
  - Add CUDA cache clearing between rounds
  - Implement checkpoint compression
  - Add memory usage monitoring and warnings
  - _Requirements: 10.2_

- [ ] 12.3 Optimize SHAP computation
  - Limit background samples to 100-200 for efficiency
  - Implement batch processing for SHAP values
  - Compute SHAP only on test set samples (not full dataset)
  - _Requirements: 10.2_

- [ ] 13. Final integration and testing
- [ ] 13.1 Create end-to-end example workflow
  - Create example script demonstrating complete workflow from data to results
  - Test on small subset of MIMIC-CXR (1000 images, 3 clients)
  - Verify all outputs are generated correctly
  - Document expected runtime and resource requirements
  - _Requirements: 8.6_

- [ ] 13.2 Perform final validation
  - Run complete experiment with full MIMIC-CXR dataset
  - Verify reproducibility by running with same seed multiple times
  - Validate all metrics are within expected ranges
  - Check all visualizations are publication-ready
  - Verify privacy guarantees are maintained
  - _Requirements: 8.1, 8.2, 9.1_

- [ ]* 13.3 Achieve test coverage target
  - Run coverage analysis with pytest-cov
  - Identify uncovered code paths
  - Add tests to reach 70% coverage minimum
  - Document any intentionally untested code
  - _Requirements: 12.3, 12.4_
