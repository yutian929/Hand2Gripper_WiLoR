"""
Hand2Gripper WiLoR Package

A unified interface for hand detection and 3D reconstruction using WiLoR model.
"""

__version__ = "0.1.0"
__author__ = "Hand2Gripper Team"
__email__ = "your-email@example.com"

# Import main classes
from .hand2gripper_wilor import HandDetector, WiLoRModel, HandRenderer

# Make main classes available at package level
__all__ = [
    "HandDetector",
    "WiLoRModel", 
    "HandRenderer",
    "__version__",
    "__author__",
    "__email__",
]