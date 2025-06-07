"""Module with LightningDataModule classes for loading ESC50 dataset."""

from pathlib import Path

import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader

from esc50_mel_cnn.download_data import download_esc50
from esc50_mel_cnn.data import RawAudioDataset
from esc50_mel_cnn.data_transforms import AudioAug, LogMelSpectrogram


class AudioMelDataModule(L.LightningDataModule):
    """LightningDataModule for loading mel spectrograms of audio."""

    def __init__(
        self,
        dataset_path: str | Path,
        aug_rate: float = 0.75,
        batch_size: int = 8,
        num_workers: int = 8,
    ):
        """Initialize the LightningDataModule.

        Args:
            dataset_path (str | Path): path to the ESC-50 dataset directory.
            aug_rate (float, optional): probability of applying augmentation
                (on train data). Defaults to 0.75.
            batch_size (int, optional): batch size. Defaults to 8.
            num_workers (int, optional): number of workers for data loading.
        """
        super().__init__()
        self.save_hyperparameters()
        self.dataset_path = Path(dataset_path)
        self.aug_rate = aug_rate
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        """Download ESC-50 dataset if it is not already downloaded."""
        download_esc50(self.dataset_path)

    def setup(self, stage: str | None = None) -> None:
        """Prepare data for training and testing.

        Args:
            stage (str | None, optional): can be 'fit', 'validate', 'test' or
                'predict'. Defaults to None.
        """
        table_path = self.dataset_path / "esc50.csv"
        audio_path = self.dataset_path / "audio" / "audio"
        train_transform = nn.Sequential(
            AudioAug(self.aug_rate), LogMelSpectrogram()
        )
        self.train_dataset = RawAudioDataset(
            table_path, audio_path, train=True, transform=train_transform
        )
        test_transform = LogMelSpectrogram()
        self.test_dataset = RawAudioDataset(
            table_path, audio_path, train=False, transform=test_transform
        )

    def train_dataloader(self) -> DataLoader:
        """Get train dataloader. Uses shuffling and augmentations."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """Get val (test) dataloader. Uses no shuffling and augmentations."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """Get val (test) dataloader. Uses no shuffling and augmentations."""
        return self.val_dataloader()
