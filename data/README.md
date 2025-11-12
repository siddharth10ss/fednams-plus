# Data Module

This module handles dataset downloading, preprocessing, and federated partitioning for FedNAMs+.

## Components

### DataDownloader (`downloader.py`)
Downloads and validates the MIMIC-CXR dataset from Kaggle or manual sources.

**Usage:**
```python
from data import DataDownloader

downloader = DataDownloader()
data_dir = downloader.download(source='kaggle', output_dir='data/mimic-cxr')
is_valid = downloader.verify_integrity(data_dir)
```

### DataPreprocessor (`preprocessor.py`)
Preprocesses chest X-ray images with resizing, normalization, and augmentation.

**Usage:**
```python
from data import DataPreprocessor
from configs import PreprocessConfig

config = PreprocessConfig(
    image_size=(224, 224),
    normalization='imagenet',
    augmentation=True
)

preprocessor = DataPreprocessor(config)
train_ds, val_ds, test_ds = preprocessor.create_train_val_test_split(dataset)
```

### FederatedDataPartitioner (`partitioner.py`)
Partitions data into non-IID client subsets for federated learning.

**Strategies:**
- **Dirichlet**: Label distribution heterogeneity (alpha parameter)
- **Pathology**: Clients specialize in different pathologies
- **Quantity**: Varying dataset sizes across clients

**Usage:**
```python
from data import FederatedDataPartitioner

partitioner = FederatedDataPartitioner(
    num_clients=5,
    strategy='dirichlet',
    alpha=0.5
)

client_datasets = partitioner.partition(dataset)
stats = partitioner.generate_statistics(client_datasets, output_dir='outputs/partitions')
```

### BaseDataset (`base_dataset.py`)
Abstract interface for extending to new datasets.

**Adding a New Dataset:**

1. Inherit from `BaseDataset`
2. Implement required methods: `load()`, `preprocess()`, `get_labels()`, `get_label_names()`
3. Register with `@DatasetRegistry.register('dataset-name')`

**Example:**
```python
from data.base_dataset import BaseDataset, DatasetRegistry

@DatasetRegistry.register('chexpert')
class CheXpertDataset(BaseDataset):
    def load(self):
        # Load CheXpert data
        pass
    
    def preprocess(self, **kwargs):
        # Preprocess CheXpert
        pass
    
    def get_labels(self):
        return self.labels
    
    def get_label_names(self):
        return ['Cardiomegaly', 'Edema', ...]
```

## MIMIC-CXR Dataset

### Structure
```
data/mimic-cxr/
├── files/                              # Image files
│   ├── p10/
│   ├── p11/
│   └── ...
├── mimic-cxr-2.0.0-chexpert.csv       # Labels
└── mimic-cxr-2.0.0-metadata.csv       # Metadata
```

### Labels
14 pathology classes:
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Enlarged Cardiomediastinum
- Fracture
- Lung Lesion
- Lung Opacity
- No Finding
- Pleural Effusion
- Pleural Other
- Pneumonia
- Pneumothorax
- Support Devices

## Extending to Other Datasets

### CheXpert
```python
# Similar structure to MIMIC-CXR
# 5 pathology classes
# ~224k images
```

### NIH Chest X-rays
```python
# 14 pathology classes
# ~112k images
# Different label format (text-based)
```

See `base_dataset.py` for the interface to implement.
