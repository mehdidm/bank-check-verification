"""
Bank Check Processing Package
Provides tools for check image preprocessing, OCR extraction, and validation
"""

# Import key classes for easy access
from .preprocessing import CheckPreprocessor
from .ocr_extraction import CheckProcessor
from .annotation_utils import LabelStudioAnnotator
from .check_validator import CheckValidator

# Define what gets imported with *
__all__ = [
    'CheckPreprocessor',
    'CheckProcessor',
    'LabelStudioAnnotator',
    'CheckValidator'
]

# Package version
__version__ = '0.1.0'