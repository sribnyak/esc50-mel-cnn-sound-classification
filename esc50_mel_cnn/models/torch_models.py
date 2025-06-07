"""This module contains model architectures used for classification.

It includes a CNN and a simple MLP classifier used as a baseline.
Models are implemented here as torch.nn.Module classes."""

import torch
import torch.nn as nn


class SimpleMLPClassifier(nn.Module):
    """Simple MLP classifier with one hidden layer."""

    def __init__(
        self, num_features: int, num_classes: int, hidden_dim: int = 100
    ):
        """Initialize the model.

        Args:
            num_features (int): number of input features.
            num_classes (int): number of output classes.
            hidden_dim (int, optional): number of neurons in the hidden layer.
                Defaults to 100.
        """
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): batch of input features,
                shape: (batch_size, num_features).

        Returns:
            torch.Tensor: logits, shape: (batch_size, num_classes)
        """
        return self.model(x)


class CNNClassifier(nn.Module):
    """CNN classifier with one linear layer.

    Designed for mel-spectrograms of size 128 x 431, but can be used for any
    one-channel images. It uses 4 average poolings 2x2 and one global average
    pooling, so the image size should preferably be greater than 16 x 16."""

    def __init__(self, num_classes: int):
        """Initialize the model.

        Args:
            num_classes (int): number of output classes.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(8, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): batch of input images.
                Shape: (batch_size, 1, height, width)

        Returns:
            torch.Tensor: batch of logits, shape: (batch_size, num_classes).
        """
        return self.model(x)
