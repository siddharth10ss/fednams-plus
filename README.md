# FedNAMs+: Interpretable Federated Neural Additive Models

**FedNAMs+** is an interpretable federated learning system for medical imaging that combines Neural Additive Models (NAMs), SHAP explanations, and uncertainty quantification for chest X-ray analysis.

## Overview

This system enables privacy-preserving collaborative learning across multiple simulated hospital clients while providing transparent, client-specific explanations and uncertainty estimates using the public MIMIC-CXR dataset.

## Features

- **Neural Additive Models (NAMs)**: Inherently interpretable architecture with additive feature contributions
- **Federated Learning**: Privacy-preserving training across distributed clients
- **SHAP Explanations**: Per-client feature importance analysis
- **Uncertainty Quantification**: Conformal prediction for reliable predictions
- **Modular Design**: Easy to extend with new models, datasets, or explainability methods

## Project Structure

```
fednams-plus/
├── data/                   # Data downloading, preprocessing, partitioning
├── models/                 # NAM architecture and baseline models
├── training/               # Local training and federated aggregation
├── explainability/         # SHAP computation and visualization
├── experiments/            # Experiment orchestration and evaluation
├── configs/                # Configuration dataclasses
├── utils/                  # Utilities, exceptions, logging
├── scripts/                # Standalone scripts for experiments
├── notebooks/              # Jupyter notebooks for Colab
├── tests/                  # Unit and integration tests
└── .kiro/specs/           # Requirements, design, and tasks

```

## Installation

### Local Setup (VS Code)

```bash
# Clone repository
git clone https://github.com/yourusername/fednams-plus.git
cd fednams-plus

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Setup

```python
# In Colab notebook
!git clone https://github.com/yourusername/fednams-plus.git
%cd fednams-plus
!pip install -r requirements.txt

# Mount Google Drive for data storage
from google.colab import drive
drive.mount('/content/drive')
```

## Quick Start

### 1. Download and Prepare Data (Colab)

```python
# Run in Colab - downloads and preprocesses MIMIC-CXR
from data import DataDownloader, DataPreprocessor, FederatedDataPartitioner

# Download data
downloader = DataDownloader()
data_dir = downloader.download(source='kaggle', output_dir='data/mimic-cxr')

# Preprocess and partition
preprocessor = DataPreprocessor()
partitioner = FederatedDataPartitioner(num_clients=5, strategy='dirichlet')
# ... (see notebooks for complete examples)
```

### 2. Train FedNAMs+ Model (Colab)

```python
# Run in Colab - trains federated model
from experiments import ExperimentRunner
from configs import ExperimentConfig

# Load configuration
config = ExperimentConfig.from_yaml('configs/experiment.yaml')

# Run experiment
runner = ExperimentRunner(config)
results = runner.run_experiment()
```

### 3. Evaluate and Visualize (Colab)

```python
# Run in Colab - generates SHAP plots and metrics
from explainability import SHAPExplainer, SHAPVisualizer

# Compute SHAP values
explainer = SHAPExplainer(model, background_data)
shap_values = explainer.explain_batch(test_loader)

# Visualize
visualizer = SHAPVisualizer()
visualizer.plot_summary(shap_values, save_path='outputs/shap_summary.png')
```

## Workflow: VS Code ↔ Colab

**Local (VS Code)**: Write modular code, utilities, configurations
**Colab**: Run training, experiments, SHAP computation, generate plots

1. Develop code locally in VS Code
2. Push to GitHub
3. Clone/pull in Colab
4. Run heavy computations in Colab
5. Download results and integrate back locally

## Configuration

All experiments are configured via YAML files in `configs/`. Example:

```yaml
experiment_name: "fednams_plus_baseline"
seed: 42
device: "cuda"

data:
  dataset: "mimic-cxr"
  num_clients: 5
  partition_strategy: "dirichlet"

model:
  backbone: "resnet18"
  pretrained: true
  feature_dim: 512

training:
  num_rounds: 100
  batch_size: 32
  learning_rate: 0.001
```

## Development Status

This project is under active development. See `.kiro/specs/fednams-plus/tasks.md` for implementation progress.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- MIMIC-CXR dataset access (Kaggle or PhysioNet)

## Citation

If you use this code, please cite:

```bibtex
@software{fednams_plus,
  title={FedNAMs+: Interpretable Federated Neural Additive Models},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/fednams-plus}
}
```

## License

[Your chosen license]

## Acknowledgments

- MIMIC-CXR dataset from PhysioNet
- PyTorch and torchvision teams
- SHAP library authors
