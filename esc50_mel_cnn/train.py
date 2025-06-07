"""The train script."""

import torch
import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    DeviceStatsMonitor,
    ModelCheckpoint,
)

from esc50_mel_cnn.data import AudioMelDataModule
from esc50_mel_cnn.models import CNNClassifier, MelClassifier

dataset_path = "./datasets/ESC-50/"  # TODO: use config
logs_path = "./logs/"  # TODO: use config
checkpoints_path = "./checkpoints/"  # TODO: use config


# TODO add dvc, logging, config, cli


def main() -> None:
    """Train the model."""
    L.seed_everything(17)
    datamodule = AudioMelDataModule(dataset_path=dataset_path)
    model = MelClassifier(
        CNNClassifier(num_classes=50),  # TODO: use config?
    )

    logger = CSVLogger(save_dir=logs_path, name="try_lightning")
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        DeviceStatsMonitor(),
        ModelCheckpoint(
            dirpath=checkpoints_path,
            filename="{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=3,
            every_n_epochs=1,
        ),
    ]

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = [1]  # torch.cuda.device_count()
    else:
        accelerator = "cpu"
        devices = 1

    trainer = Trainer(
        max_epochs=2,  # 80
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
