"""Module for loading models."""

from .torch_models import SimpleMLPClassifier, CNNClassifier
from .lightning_modules import TabularClassifier, MelClassifier

__all__ = [
    "SimpleMLPClassifier",
    "CNNClassifier",
    "TabularClassifier",
    "MelClassifier",
]
