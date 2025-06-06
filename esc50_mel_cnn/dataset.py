from pathlib import Path

import pandas as pd
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, table_path, audio_path, train, transform=None):
        self.train = train
        self.transform = transform

        table = pd.read_csv(Path(table_path) / "esc50.csv")
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

    def __getitem__(self, index):
        item = self.loaded_data[index]
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.loaded_data)
