"""Utility module for FedNAMs+ system.

This module provides common utilities, exceptions, and logging functionality.
"""

from .exceptions import (
    FedNAMsError,
    DataError,
    ModelError,
    TrainingError,
    ExplanationError,
    ConfigurationError
)
from .logging_utils import setup_logger, get_logger

__all__ = [
    'FedNAMsError',
    'DataError',
    'ModelError',
    'TrainingError',
    'ExplanationError',
    'ConfigurationError',
    'setup_logger',
    'get_logger'
]
