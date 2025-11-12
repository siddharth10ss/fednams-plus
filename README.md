# FedNAMs+: Federated Neural Additive Models with Explainability and Uncertainty Quantification

A privacy-preserving federated learning framework for medical image classification that combines interpretable Neural Additive Models (NAMs) with SHAP-based explanations and conformal prediction for uncertainty quantification.

## Overview

FedNAMs+ enables collaborative training of interpretable deep learning models across multiple healthcare institutions without sharing sensitive patient data. The framework provides:

- **Privacy-Preserving Federated Learning**: Train models collaboratively while keeping data decentralized
- **Built-in Interpretability**: Neural Additive Models provide feature-level explanations
- **Post-hoc Explainability**: SHAP values for detailed feature importance analysis
- **Uncertainty Quantification**: Conformal prediction for reliable confidence estimates
- **Medical Image Focus**: Optimized for chest X-ray classification (MIMIC-CXR, NIH ChestX-ray14)

## Key Features

- Federated learning with FedAvg aggregation
- Multiple CNN backbones (ResNet18, ResNet50, DenseNet121)
- Neural Additive Model architecture for interpretability
- SHAP-based explanation analysis and visualization
- Adaptive Prediction Sets for uncertainty quantification
- Support for non-IID data distributions
- Comprehensive evaluation metrics
- Optional differential privacy (DP-SGD)

## Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fednams-plus.git
cd fednams-plus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Installation

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/yourusername/fednams-plus.git
%cd fednams-plus

# Install dependencies
!pip install -r requirements.txt
```

## Quick Start

### Basic Training

```bash
# Run experiment with default configuration
python scripts/run_experiment.py --config configs/experiment.yaml

# Override specific parameters
python scripts/run_experiment.py \
    --config configs/experiment.yaml \
    --device cuda \
    --seed 42
```

### Using Python API

```python
from pathlib import Path
from configs.config_loader import ConfigLoader
from experiments import ExperimentRunner

# Load configuration
config = ConfigLoader.load_experiment_config(Path("configs/experiment.yaml"))

# Create and run experiment
runner = ExperimentRunner(config)
results = runner.run_experiment()

# Save results
runner.save_results(results)
```

## Configuration

The framework uses YAML configuration files. Key configuration sections:

### Data Configuration

```yaml
data:
  dataset: "nih-cxr"
  data_dir: "/path/to/data"
  image_size: [224, 224]
  normalization: "imagenet"
  augmentation: true
```

### Model Configuration

```yaml
model:
  backbone: "resnet18"  # resnet18, resnet50, densenet121
  pretrained: true
  feature_dim: 512
  num_classes: 15
  nam_hidden_units: [64, 32]
  dropout: 0.3
```

### Federated Learning Configuration

```yaml
federated:
  num_clients: 5
  num_rounds: 50
  client_fraction: 1.0
  min_clients: 3
  partition_strategy: "dirichlet"
```

### Training Configuration

```yaml
training:
  batch_size: 32
  learning_rate: 0.001
  num_local_epochs: 5
  optimizer: "adam"
  scheduler: "cosine"
  mixed_precision: true
```

See `configs/experiment.yaml` for complete configuration options.

## Project Structure

```
fednams-plus/
├── configs/              # Configuration files
│   ├── config.py        # Configuration dataclasses
│   ├── config_loader.py # YAML loader
│   └── experiment.yaml  # Default experiment config
├── data/                # Data handling
│   ├── downloader.py    # Dataset downloading
│   ├── preprocessor.py  # Image preprocessing
│   └── partitioner.py   # Federated data partitioning
├── models/              # Model architectures
│   ├── feature_extractor.py  # CNN backbones
│   ├── nam_head.py      # NAM architecture
│   ├── fednams_model.py # Complete model
│   └── baselines.py     # Baseline models
├── training/            # Training components
│   ├── local_trainer.py # Client-side training
│   ├── aggregator.py    # FedAvg aggregation
│   └── orchestrator.py  # Federated coordination
├── explainability/      # Explanation modules
│   ├── shap_explainer.py     # SHAP computation
│   ├── shap_visualizer.py    # Visualization
│   └── explanation_analyzer.py # Quality metrics
├── uncertainty/         # Uncertainty quantification
│   └── conformal_predictor.py # Conformal prediction
├── experiments/         # Experiment management
│   ├── experiment_runner.py  # Main runner
│   └── evaluation_metrics.py # Metrics computation
├── scripts/             # CLI scripts
│   └── run_experiment.py     # Main entry point
├── utils/               # Utilities
│   ├── exceptions.py    # Custom exceptions
│   └── logging_utils.py # Logging setup
└── docs/                # Documentation
    └── PRIVACY.md       # Privacy documentation
```

## Usage Examples

### Running Baseline Comparisons

```bash
# FedNAMs+ (main approach)
python scripts/run_experiment.py --config configs/experiment.yaml

# FedAvg CNN baseline
python scripts/run_experiment.py --config configs/baselines.yaml

# Centralized NAM (privacy baseline)
python scripts/run_experiment.py --config configs/centralized.yaml
```

### Resuming from Checkpoint

```bash
python scripts/run_experiment.py \
    --config configs/experiment.yaml \
    --resume outputs/experiment_name/checkpoints/round_25.pt
```

### Evaluation Only

```python
from experiments import ExperimentRunner

runner = ExperimentRunner(config)
runner.load_checkpoint("path/to/checkpoint.pt")
results = runner.evaluate_model()
```

## Datasets

### Supported Datasets

- **NIH ChestX-ray14**: 112,120 frontal-view X-ray images with 14 disease labels
- **MIMIC-CXR**: 377,110 chest X-rays with 14 pathology labels

### Data Preparation

1. Download dataset from official source
2. Extract to data directory
3. Update `data_dir` in configuration file
4. Run preprocessing (handled automatically during training)

### Data Partitioning

The framework supports multiple partitioning strategies for federated learning:

- **Dirichlet**: Non-IID distribution using Dirichlet parameter α
- **Pathology-based**: Clients specialize in specific diseases
- **Quantity-based**: Varying dataset sizes across clients

## Evaluation Metrics

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, AUC-PR (per-class and macro-averaged)

### Explanation Metrics
- SHAP consistency (repeated computation agreement)
- Feature stability (robustness to perturbations)
- Cross-client feature agreement

### Uncertainty Metrics
- Coverage (empirical vs target confidence)
- Average prediction set size
- Conditional coverage per class

### Communication Metrics
- Parameters transmitted per round
- Total communication cost (MB)

## Interpretability and Explainability

### NAM Feature Contributions

```python
# Get feature-level contributions
model = runner.model
predictions, contributions = model.forward_with_contributions(images)

# contributions shape: [batch_size, feature_dim, num_classes]
```

### SHAP Analysis

```python
from explainability import SHAPExplainer, SHAPVisualizer

# Compute SHAP values
explainer = SHAPExplainer(model, background_data)
shap_values = explainer.explain_batch(test_data)

# Visualize
visualizer = SHAPVisualizer()
visualizer.plot_summary(shap_values, output_dir)
visualizer.plot_feature_importance(shap_values, output_dir)
```

### Uncertainty Quantification

```python
from uncertainty import ConformalPredictor

# Calibrate predictor
predictor = ConformalPredictor(confidence_level=0.9)
predictor.calibrate(model, calibration_data)

# Get prediction sets
prediction_sets = predictor.predict_with_sets(test_data)
```

## Privacy and Security

FedNAMs+ implements multiple privacy safeguards:

- **Data Localization**: Raw data never leaves client institutions
- **Parameter-Only Communication**: Only model weights are transmitted
- **Audit Logging**: All data access and transmissions are logged
- **Optional Differential Privacy**: DP-SGD with privacy budget tracking

See [docs/PRIVACY.md](docs/PRIVACY.md) for detailed privacy documentation.

## Troubleshooting

### CUDA Out of Memory

```yaml
# Reduce batch size
training:
  batch_size: 16  # or 8

# Enable gradient accumulation
training:
  gradient_accumulation_steps: 2
```

### Slow SHAP Computation

```yaml
# Reduce background samples
explainability:
  shap_background_samples: 50  # default: 100
  shap_test_samples: 200       # default: 500
```

### Convergence Issues

```yaml
# Adjust learning rate
training:
  learning_rate: 0.0001  # reduce if unstable

# Enable early stopping
training:
  early_stopping_patience: 10
```

## Citation

If you use FedNAMs+ in your research, please cite:

```bibtex
@article{fednams_plus,
  title={FedNAMs+: Privacy-Preserving Federated Learning with Interpretability and Uncertainty Quantification for Medical Imaging},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Neural Additive Models (NAMs) architecture
- SHAP (SHapley Additive exPlanations) library
- Conformal Prediction framework
- PyTorch and torchvision communities

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].
