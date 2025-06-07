"""Module for loading ESC50 dataset."""

from pathlib import Path

import pandas as pd
import torch
import torchaudio  # type: ignore
from torch.utils.data import Dataset

from typing import Callable


class RawAudioDataset(Dataset):
    """Raw audio dataset. Returns tuples (raw audio, sample rate, label)."""

    def __init__(
        self,
        table_path: str | Path,
        audio_path: str | Path,
        train: bool,
        transform: Callable | None = None,
    ):
        """Initialize the dataset.

        Args:
            table_path (str | Path): path to the table with metadata.
            audio_path (str | Path): path to the directory with audio files.
            train (bool): whether to load train or test data.
            transform (Callable | None, optional): transform to apply to
                the data. Defaults to None.
        """
        self.train = train
        self.transform = transform

        table = pd.read_csv(table_path)
        self.num_categories = len(table["category"].unique())

        labels = {i: s for i, s in zip(table["target"], table["category"])}
        self.labels = [labels[i] for i in range(self.num_categories)]

        train_mask = table["fold"] < 5
        table = table[train_mask] if self.train else table[~train_mask]

        self.loaded_data = []
        for i, row in table.iterrows():
            path = Path(audio_path) / row["filename"]
            label = row["target"]
            data_tensor, rate = torchaudio.load(path)
            self.loaded_data.append((data_tensor, rate, label))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        """Get data item at index, transform if necessary.

        Args:
            index (int): index of the item in the dataset.

        Returns:
            tuple[torch.Tensor, int, int]: audio data, sample rate, label.
        """
        item = self.loaded_data[index]
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.loaded_data)


# TODO TabularAudioDataset
