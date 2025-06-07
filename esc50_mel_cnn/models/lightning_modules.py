"""This module contains LightningModule wrappers for the models used."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from typing import Any  # TODO: maybe not needed


class TabularClassifier(L.LightningModule):
    """LightningModule wrapper for a tabular classifier (NLP in our case)."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
    ):
        """Initialize the LightningModule.

        Args:
            model (nn.Module): model to use. (Expected NLP, but any alternative
                with the same input and output format can be used)
            learning_rate (float, optional): learning rate. Defaults to 1e-3.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): batch of input features,
                shape: (batch_size, num_features).

        Returns:
            torch.Tensor: logits, shape: (batch_size, num_classes)
        """
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch (Any): batch of data, expected to be a tuple of (x, y).
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: loss as a differentiable tensor.
        """
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch (Any): batch of data, expected to be a tuple of (x, y).
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: loss as a tensor.
        """
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Test step.

        Args:
            batch (Any): batch of data, expected to be a tuple of (x, y).
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: loss as a tensor.
        """
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer.

        Returns:
            torch.optim.Optimizer: optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class MelClassifier(TabularClassifier):  # TODO remove if unnecessary
    """LightningModule wrapper for a classifier that uses a mel-spectrogram.

    In our case, the model is a CNN.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 3e-4,
    ):
        """Initialize the LightningModule.

        Args:
            model (nn.Module): model to use. (Expected CNN, but any alternative
                with the same input and output format can be used)
            learning_rate (float, optional): learning rate. Defaults to 1e-3.
        """
        super().__init__(model, learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): batch of input images.
                Shape: (batch_size, 1, height, width)

        Returns:
            torch.Tensor: batch of logits, shape: (batch_size, num_classes).
        """
        return self.model(x)
