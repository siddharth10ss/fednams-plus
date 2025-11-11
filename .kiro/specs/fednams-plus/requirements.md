# Requirements Document

## Introduction

FedNAMs+ is an interpretable federated learning system for medical imaging that combines Neural Additive Models (NAMs), SHAP explanations, and uncertainty quantification for chest X-ray analysis. The system enables privacy-preserving collaborative learning across multiple simulated hospital clients while providing transparent, client-specific explanations and uncertainty estimates using the public MIMIC-CXR dataset.

## Glossary

- **FedNAMs+ System**: The complete federated learning framework including NAM models, SHAP explainability, and conformal prediction components
- **Client**: A simulated hospital node in the federated network that trains locally on its data partition
- **Server**: The central aggregation component that coordinates federated training without accessing raw data
- **NAM (Neural Additive Model)**: An interpretable neural network architecture that learns additive feature contributions
- **SHAP (SHapley Additive exPlanations)**: A method for computing feature importance values for model predictions
- **Conformal Prediction**: A framework for quantifying prediction uncertainty with statistical guarantees
- **MIMIC-CXR Dataset**: The publicly available chest X-ray dataset from PhysioNet/Kaggle used for training and evaluation
- **Feature Extractor**: A CNN component that transforms raw images into feature representations for the NAM
- **Global Model**: The aggregated model parameters shared across all clients after federated averaging
- **Local Model**: Client-specific model parameters before aggregation
- **Non-IID Split**: A data partitioning strategy where different clients have heterogeneous data distributions

## Requirements

### Requirement 1: Data Preparation and Federated Environment Setup

**User Story:** As a researcher, I want to prepare the MIMIC-CXR dataset and simulate a federated environment, so that I can train models across multiple virtual hospital clients with realistic data heterogeneity.

#### Acceptance Criteria

1. THE FedNAMs+ System SHALL download and preprocess the MIMIC-CXR dataset from Kaggle or PhysioNet sources
2. THE FedNAMs+ System SHALL partition the preprocessed dataset into at least 3 client subsets using non-IID splitting strategies
3. THE FedNAMs+ System SHALL store each client's data partition separately to simulate federated data isolation
4. THE FedNAMs+ System SHALL generate metadata files documenting the data distribution statistics for each client partition
5. THE FedNAMs+ System SHALL validate that all image files are properly formatted and accessible before training begins

### Requirement 2: Neural Additive Model Architecture

**User Story:** As a machine learning engineer, I want to implement NAM architecture for medical images, so that the model provides inherently interpretable predictions through additive feature contributions.

#### Acceptance Criteria

1. THE FedNAMs+ System SHALL implement a Feature Extractor component using convolutional neural networks to transform chest X-ray images into feature vectors
2. THE FedNAMs+ System SHALL implement a NAM head component that processes extracted features through separate neural networks for each feature dimension
3. THE FedNAMs+ System SHALL combine individual feature network outputs additively to produce final predictions
4. THE FedNAMs+ System SHALL support multi-class classification for common chest X-ray pathologies
5. THE FedNAMs+ System SHALL expose individual feature contribution functions for interpretability analysis

### Requirement 3: Federated Training Infrastructure

**User Story:** As a federated learning researcher, I want to implement federated training with secure aggregation, so that multiple clients can collaboratively train models without sharing raw medical data.

#### Acceptance Criteria

1. THE FedNAMs+ System SHALL implement local training routines that train the NAM model on each client's private data partition
2. WHEN a training round completes, THE FedNAMs+ System SHALL aggregate client model parameters using the FedAvg algorithm on the Server
3. THE FedNAMs+ System SHALL distribute the Global Model parameters to all clients after each aggregation round
4. THE FedNAMs+ System SHALL track and log training metrics including loss and accuracy for each client and round
5. THE FedNAMs+ System SHALL support configurable hyperparameters including learning rate, batch size, local epochs, and communication rounds
6. THE FedNAMs+ System SHALL ensure that raw image data never leaves the client during training or aggregation

### Requirement 4: Local SHAP Explanation Generation

**User Story:** As a clinician, I want to receive SHAP-based explanations for predictions made on my hospital's data, so that I can understand which image features influenced the diagnosis and validate model decisions.

#### Acceptance Criteria

1. WHEN a client completes local training, THE FedNAMs+ System SHALL compute SHAP values for a representative sample of that client's test data
2. THE FedNAMs+ System SHALL generate SHAP summary plots showing feature importance distributions for each client
3. THE FedNAMs+ System SHALL store SHAP values and visualizations in client-specific output directories
4. THE FedNAMs+ System SHALL support both global feature importance aggregation and individual prediction explanations
5. THE FedNAMs+ System SHALL compute SHAP value statistics including mean absolute values and variance across samples

### Requirement 5: Uncertainty Quantification with Conformal Prediction

**User Story:** As a healthcare provider, I want uncertainty estimates for each prediction, so that I can assess prediction reliability and make informed clinical decisions.

#### Acceptance Criteria

1. THE FedNAMs+ System SHALL implement conformal prediction calibration using a held-out calibration set from each client
2. WHEN making predictions, THE FedNAMs+ System SHALL generate prediction sets with user-specified confidence levels
3. THE FedNAMs+ System SHALL compute and report coverage metrics comparing actual coverage to target confidence levels
4. THE FedNAMs+ System SHALL provide prediction interval widths as a measure of uncertainty magnitude
5. THE FedNAMs+ System SHALL support both client-specific and global conformal prediction models

### Requirement 6: Baseline Comparison and Evaluation

**User Story:** As a researcher, I want to compare FedNAMs+ against standard federated learning baselines, so that I can demonstrate the value of interpretability and uncertainty quantification.

#### Acceptance Criteria

1. THE FedNAMs+ System SHALL implement baseline models including FedAvg with CNN, FedAvg with GradCAM, and centralized NAM
2. THE FedNAMs+ System SHALL evaluate all models using standard metrics including AUC, F1-score, accuracy, precision, and recall
3. THE FedNAMs+ System SHALL measure and compare explanation quality metrics including SHAP consistency and feature stability
4. THE FedNAMs+ System SHALL quantify communication costs by tracking the number of parameters transmitted per round
5. THE FedNAMs+ System SHALL generate comparison tables and visualizations for all evaluation metrics across models

### Requirement 7: Modular and Extensible Codebase

**User Story:** As a developer, I want a well-organized modular codebase, so that I can easily extend the system with new models, explainability methods, or datasets.

#### Acceptance Criteria

1. THE FedNAMs+ System SHALL organize code into separate modules for data processing, models, training, explainability, and experiments
2. THE FedNAMs+ System SHALL define clear interfaces for swapping between local, federated, and centralized training modes
3. THE FedNAMs+ System SHALL provide configuration files for specifying experiment parameters without code modification
4. THE FedNAMs+ System SHALL include comprehensive docstrings for all public functions and classes
5. THE FedNAMs+ System SHALL support dependency management through a requirements.txt file listing all Python packages

### Requirement 8: Reproducible Experimentation and Visualization

**User Story:** As a researcher preparing a publication, I want reproducible experiments with comprehensive visualizations, so that I can validate results and communicate findings effectively.

#### Acceptance Criteria

1. THE FedNAMs+ System SHALL set random seeds for all stochastic components to ensure reproducibility
2. THE FedNAMs+ System SHALL save model checkpoints, training logs, and evaluation metrics for each experiment run
3. THE FedNAMs+ System SHALL generate publication-ready plots including training curves, SHAP visualizations, and comparison charts
4. THE FedNAMs+ System SHALL export results in standard formats including CSV for metrics and PNG for visualizations
5. THE FedNAMs+ System SHALL provide Jupyter notebooks demonstrating end-to-end workflows for Google Colab execution
6. THE FedNAMs+ System SHALL include README documentation with setup instructions, usage examples, and experiment reproduction steps

### Requirement 9: Privacy and Security Considerations

**User Story:** As a privacy officer, I want assurance that the federated system protects patient data, so that the system complies with healthcare privacy regulations.

#### Acceptance Criteria

1. THE FedNAMs+ System SHALL ensure that only model parameters are transmitted between clients and the Server during federated training
2. THE FedNAMs+ System SHALL provide documentation of privacy guarantees and potential privacy risks
3. THE FedNAMs+ System SHALL support optional differential privacy mechanisms for gradient perturbation
4. THE FedNAMs+ System SHALL log all data access and model transmission events for audit purposes
5. THE FedNAMs+ System SHALL document SHAP explanation privacy properties based on established literature and validate that explanations do not directly expose individual training samples

### Requirement 10: Performance and Scalability

**User Story:** As a system administrator, I want the system to train efficiently on available hardware, so that experiments can be completed within reasonable timeframes.

#### Acceptance Criteria

1. WHERE GPU acceleration is available, THE FedNAMs+ System SHALL utilize GPU resources for model training and inference
2. THE FedNAMs+ System SHALL support batch processing for SHAP value computation to manage memory usage
3. THE FedNAMs+ System SHALL implement early stopping mechanisms to prevent unnecessary training iterations
4. THE FedNAMs+ System SHALL provide progress indicators and estimated time remaining during long-running operations
5. THE FedNAMs+ System SHALL support configurable client sampling to reduce communication overhead in large-scale scenarios

### Requirement 11: Dataset Extensibility

**User Story:** As a researcher, I want to easily adapt the system to other medical imaging datasets, so that I can apply the FedNAMs+ approach to different clinical problems without major code rewrites.

#### Acceptance Criteria

1. THE FedNAMs+ System SHALL define abstract data loader interfaces that separate dataset-specific logic from core training code
2. THE FedNAMs+ System SHALL provide example implementations for MIMIC-CXR and document the steps to add new datasets
3. THE FedNAMs+ System SHALL support configurable image preprocessing pipelines through configuration files
4. THE FedNAMs+ System SHALL allow specification of dataset-specific label mappings and class definitions
5. THE FedNAMs+ System SHALL include documentation with guidelines for adapting the system to datasets such as CheXpert or NIH Chest X-rays

### Requirement 12: Testing and Quality Assurance

**User Story:** As a developer, I want comprehensive automated tests, so that I can confidently modify the codebase and ensure system reliability for potential open-source release.

#### Acceptance Criteria

1. THE FedNAMs+ System SHALL include unit tests for all data processing, model, and utility functions
2. THE FedNAMs+ System SHALL include integration tests that validate end-to-end training workflows
3. THE FedNAMs+ System SHALL achieve at least 70 percent code coverage across core modules
4. THE FedNAMs+ System SHALL provide a test suite that can be executed with a single command
5. THE FedNAMs+ System SHALL include continuous integration configuration for automated test execution

### Requirement 13: User Interface and Monitoring

**User Story:** As an operator, I want a simple interface to monitor training progress and manage experiments, so that I can efficiently run and track multiple experimental configurations.

#### Acceptance Criteria

1. THE FedNAMs+ System SHALL provide a command-line interface for launching experiments with configurable parameters
2. THE FedNAMs+ System SHALL display real-time training progress including current round, loss, and accuracy metrics
3. THE FedNAMs+ System SHALL support experiment management commands for listing, resuming, and terminating training runs
4. WHERE a web dashboard is implemented, THE FedNAMs+ System SHALL provide visualization of training curves and client statistics
5. THE FedNAMs+ System SHALL log all experiment outputs to structured log files for post-hoc analysis
