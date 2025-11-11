"""Custom exceptions for FedNAMs+ system."""


class FedNAMsError(Exception):
    """Base exception for FedNAMs+ system.
    
    All custom exceptions in the system inherit from this base class.
    """
    pass


class DataError(FedNAMsError):
    """Errors related to data loading and preprocessing.
    
    Raised when:
    - Dataset files are missing or corrupted
    - Data preprocessing fails
    - Data partitioning encounters issues
    """
    pass


class ModelError(FedNAMsError):
    """Errors related to model architecture and operations.
    
    Raised when:
    - Model initialization fails
    - Forward pass encounters issues
    - Parameter serialization/deserialization fails
    """
    pass


class TrainingError(FedNAMsError):
    """Errors during training and aggregation.
    
    Raised when:
    - Training loop encounters NaN/Inf losses
    - Aggregation fails due to parameter mismatches
    - Checkpoint loading/saving fails
    """
    pass


class ExplanationError(FedNAMsError):
    """Errors during SHAP computation or visualization.
    
    Raised when:
    - SHAP value computation fails
    - Visualization generation encounters issues
    - Background data selection is invalid
    """
    pass


class ConfigurationError(FedNAMsError):
    """Errors in configuration or setup.
    
    Raised when:
    - Configuration validation fails
    - Required parameters are missing
    - Invalid parameter combinations are detected
    """
    pass
