"""
Selectors package for managing and generating CSS/XPath selectors.

This package provides utilities for working with web element selectors,
including generation, validation and optimization of selectors.
"""

from .selector_manager import SelectorManager

__version__ = "0.1.0"
__author__ = "Bhavana Kedari"

# Export main classes
__all__ = ["SelectorManager"]