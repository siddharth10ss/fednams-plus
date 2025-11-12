"""Data downloader for MIMIC-CXR dataset."""

import os
import hashlib
import zipfile
from pathlib import Path
from typing import Optional
import subprocess

from utils.exceptions import DataError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class DataDownloader:
    """Downloads and validates the MIMIC-CXR dataset.
    
    Supports downloading from:
    - Kaggle API
    - Manual download (user provides path)
    """
    
    def __init__(self):
        """Initialize the data downloader."""
        self.kaggle_dataset = "mimic-cxr-jpg"
        
    def download(self, source: str, output_dir: str) -> Path:
        """Download the MIMIC-CXR dataset.
        
        Args:
            source: Download source ('kaggle' or 'manual')
            output_dir: Directory to save downloaded data
            
        Returns:
            Path to downloaded data directory
            
        Raises:
            DataError: If download fails or source is invalid
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if source == 'kaggle':
            return self._download_from_kaggle(output_path)
        elif source == 'manual':
            logger.info(f"Manual download mode. Please place MIMIC-CXR data in: {output_path}")
            return output_path
        else:
            raise DataError(f"Invalid download source: {source}. Use 'kaggle' or 'manual'")
    
    def _download_from_kaggle(self, output_path: Path) -> Path:
        """Download dataset using Kaggle API.
        
        Args:
            output_path: Directory to save data
            
        Returns:
            Path to downloaded data
            
        Raises:
            DataError: If Kaggle API is not configured or download fails
        """
        try:
            import kaggle
        except ImportError:
            raise DataError(
                "Kaggle package not installed. Install with: pip install kaggle"
            )
        
        # Check for Kaggle credentials
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        if not kaggle_json.exists():
            raise DataError(
                f"Kaggle credentials not found at {kaggle_json}. "
                "Please set up Kaggle API credentials: "
                "https://github.com/Kaggle/kaggle-api#api-credentials"
            )
        
        logger.info(f"Downloading MIMIC-CXR from Kaggle to {output_path}")
        
        try:
            # Download using Kaggle API
            # Note: Adjust dataset name based on actual Kaggle dataset
            kaggle.api.dataset_download_files(
                'mimic-cxr-jpg',
                path=str(output_path),
                unzip=True
            )
            logger.info("Download completed successfully")
            return output_path
            
        except Exception as e:
            raise DataError(f"Failed to download from Kaggle: {str(e)}")
    
    def verify_integrity(self, data_dir: Path) -> bool:
        """Verify the integrity of downloaded data.
        
        Args:
            data_dir: Directory containing the data
            
        Returns:
            True if data is valid, False otherwise
        """
        logger.info(f"Verifying data integrity in {data_dir}")
        
        # Check if directory exists
        if not data_dir.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            return False
        
        # Check for required files/directories
        required_items = [
            'files',  # Image files directory
            'mimic-cxr-2.0.0-chexpert.csv',  # Labels file
            'mimic-cxr-2.0.0-metadata.csv',  # Metadata file
        ]
        
        missing_items = []
        for item in required_items:
            item_path = data_dir / item
            if not item_path.exists():
                missing_items.append(item)
                logger.warning(f"Missing required item: {item}")
        
        if missing_items:
            logger.error(f"Data integrity check failed. Missing: {missing_items}")
            return False
        
        # Count image files
        image_dir = data_dir / 'files'
        if image_dir.exists():
            image_count = sum(1 for _ in image_dir.rglob('*.jpg'))
            logger.info(f"Found {image_count} image files")
            
            if image_count == 0:
                logger.error("No image files found")
                return False
        
        logger.info("Data integrity check passed")
        return True
    
    def compute_checksum(self, file_path: Path, algorithm: str = 'md5') -> str:
        """Compute checksum of a file.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm ('md5', 'sha256')
            
        Returns:
            Hexadecimal checksum string
        """
        if algorithm == 'md5':
            hasher = hashlib.md5()
        elif algorithm == 'sha256':
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
