# Data Module

This module handles dataset management for FedNAMs+, including downloading, preprocessing, and federated partitioning.

## Supported Datasets

### NIH Chest X-ray Dataset (Primary)
- **Source**: https://www.kaggle.com/datasets/nih-chest-xrays/data
- **Images**: 112,120 frontal-view chest X-rays
- **Classes**: 15 (14 diseases + No Finding)
- **Format**: PNG images, CSV labels

**Classes:**
- Atelectasis
- Cardiomegaly
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- Pleural_Thickening
- Hernia
- No Finding

### MIMIC-CXR (Alternative)
- Support exists but NIH is the primary dataset

## Usage

### Download Dataset

```python
from data import DataDownloader

downloader = DataDownloader()

# Download from Kaggle (requires kaggle.json credentials)
data_path = downloader.download(source='kaggle', output_dir='./datasets/nih-cxr')

# Or use manual download
data_path = downloader.download(source='manual', output_dir='/path/to/existing/data')

# Verify integrity
is_valid = downloader.verify_integrity(data_path)
```

### Load Dataset

```python
from data import NIHChestXrayDataset, DatasetRegistry
from pathlib import Path

# Method 1: Direct instantiation
dataset = NIHChestXrayDataset(data_dir=Path('./datasets/nih-cxr'))
dataset.load()

# Method 2: Using registry
DatasetClass = DatasetRegistry.get('nih-cxr')
dataset = DatasetClass(data_dir=Path('./datasets/nih-cxr'))
dataset.load()

# Get statistics
stats = dataset.get_statistics()
print(f"Total samples: {stats['num_samples']}")
print(f"Classes: {stats['class_names']}")
```

### Preprocess Data

```python
from data import DataPreprocessor
from configs.config import PreprocessConfig

# Configure preprocessing
config = PreprocessConfig(
    image_size=(224, 224),
    normalization='imagenet',
    augmentation=True,
    augmentation_params={
        'horizontal_flip_prob': 0.5,
        'rotation_degrees': 10,
        'brightness': 0.2,
        'contrast': 0.2
    }
)

preprocessor = DataPreprocessor(config)

# Create train/val/test splits
train_dataset, val_dataset, test_dataset = preprocessor.create_train_val_test_split(
    dataset,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
)
```

### Federated Partitioning

```python
from data import FederatedDataPartitioner

# Dirichlet partitioning (non-IID)
partitioner = FederatedDataPartitioner(
    num_clients=5,
    strategy='dirichlet',
    alpha=0.5,
    min_samples=100
)

client_datasets = partitioner.partition(train_dataset)

# Generate statistics
stats_df = partitioner.generate_statistics(
    client_datasets,
    output_dir=Path('./outputs/partitions')
)
```

## Adding a New Dataset

To add support for a new medical imaging dataset:

### 1. Create Dataset Class

Create a new file `data/your_dataset.py`:

```python
from pathlib import Path
from typing import List, Tuple
import torch
from PIL import Image

from .base_dataset import BaseDataset, DatasetRegistry
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@DatasetRegistry.register('your-dataset-name')
class YourDataset(BaseDataset):
    """Your dataset description."""
    
    def __init__(self, data_dir: Path, transform=None):
        super().__init__(data_dir)
        self.transform = transform
        self.class_names = ['Class1', 'Class2', ...]  # Define your classes
    
    def load(self) -> None:
        """Load dataset from disk."""
        # Implement loading logic:
        # 1. Find image files
        # 2. Load labels/metadata
        # 3. Populate self.image_paths and self.labels
        pass
    
    def preprocess(self, **kwargs) -> None:
        """Preprocess the dataset."""
        # Implement any dataset-specific preprocessing
        pass
    
    def get_labels(self) -> np.ndarray:
        """Get label array."""
        return self.labels
    
    def get_label_names(self) -> List[str]:
        """Get class names."""
        return self.class_names
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        # Load image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return img, label
```

### 2. Register in __init__.py

Add to `data/__init__.py`:

```python
from .your_dataset import YourDataset

__all__ = [
    ...,
    'YourDataset'
]
```

### 3. Update Downloader (Optional)

If your dataset is on Kaggle, update `data/downloader.py`:

```python
def __init__(self):
    self.kaggle_datasets = {
        'nih-cxr': 'nih-chest-xrays/data',
        'your-dataset': 'kaggle-user/dataset-name'
    }
```

### 4. Use Your Dataset

```python
from data import DatasetRegistry

# Your dataset is now available
dataset = DatasetRegistry.get('your-dataset-name')(data_dir='./data')
dataset.load()
```

## Dataset Structure

Expected directory structure for NIH Chest X-ray:

```
nih-cxr/
├── Data_Entry_2017.csv          # Labels file
├── images_001/                   # Image directory 1
│   ├── 00000001_000.png
│   ├── 00000001_001.png
│   └── ...
├── images_002/                   # Image directory 2
│   └── ...
├── ...
└── images_012/                   # Image directory 12
    └── ...
```

## Partitioning Strategies

### Dirichlet (Recommended)
- Creates non-IID distribution using Dirichlet parameter α
- Lower α = more heterogeneous
- α = 0.5 is a good default

### Pathology-based
- Clients specialize in specific diseases
- Simulates real-world hospital specializations

### Quantity-based
- Varying dataset sizes across clients
- Simulates different hospital sizes

## Notes

- All images are resized to 224x224 for model input
- ImageNet normalization is used by default
- Multi-label classification is supported
- Minimum 100 samples per client is enforced
