import random
import os  # TODO: use pathlib instead

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # TODO fix (maybe use tqdm.auto?)
from torch.utils.data import DataLoader

from esc50_mel_cnn.audio_tools import compute_log_mel_spect, random_augment
from esc50_mel_cnn.dataset import AudioDataset

# TODO change relative paths to absolute with configurable root


class LogMelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        audio, sr, target = x
        log_mel = compute_log_mel_spect(audio.numpy(), sr)
        return (torch.from_numpy(log_mel.astype(np.float32)), target)


class AudioAug(nn.Module):
    def __init__(self, p=0.75):
        super().__init__()
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return x
        else:
            audio, sr, target = x
            aug = random_augment(audio.numpy(), sr)
            return (torch.from_numpy(aug.astype(np.float32)), sr, target)


def train(model, device, train_loader, loss_criteria, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("--------------------- Epoch:", epoch, "---------------------")
    # Process the images in batches
    for data, target in tqdm(train_loader):
        # Use the CPU or GPU as appropriate
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = data.to(device), target.to(device)

        # Reset the optimizer
        optimizer.zero_grad()

        # Push the data forward through the model layers
        output = model(data.to(device))

        # Get the loss
        loss = loss_criteria(output, target)

        # Keep a running total
        train_loss += loss.item()

        # Backpropagate
        loss.backward(retain_graph=True)
        optimizer.step()

        # Print metrics so we see some progress
        # print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))

    # return average loss for the epoch
    avg_loss = train_loss / len(train_loader)
    print("Training set: Average loss: {:.6f}".format(avg_loss))
    return avg_loss


def test(model, device, test_loader, loss_criteria):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in tqdm(test_loader):
            batch_count += 1
            data, target = data.to(device), target.to(device)

            # Get the predicted classes for this batch
            output = model(data)

            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    len_dataset = len(test_loader.dataset)
    acc = 100.0 * correct / len_dataset
    print(
        f"Validation set: Average loss: {avg_loss:.6f},",
        f"Accuracy: {correct}/{len_dataset} ({acc:.0f}%)",
    )
    print()

    # return average loss for the epoch
    return avg_loss


def training(model, device, train_loader, test_loader, epochs=80,
             checkpoint_freq=10):
    # Use an "Adam" optimizer to adjust weights
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Specify the loss criteria
    loss_criteria = nn.CrossEntropyLoss()

    # Track metrics in these arrays
    epoch_nums = []
    training_loss = []
    validation_loss = []

    os.makedirs("saved_models", exist_ok=True)

    print("Training on", device)
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = train(
            model, device, train_loader, loss_criteria, optimizer, epoch
        )
        test_loss = test(model, device, test_loader, loss_criteria)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)

        if epoch % checkpoint_freq == 0:
            torch.save(model.state_dict(), f"saved_models/weights_{epoch}.pth")

    print("Done. Weights are saved to saved_models/")


def main():
    print("Creating datasets...", flush=True)
    train_dataset = AudioDataset(
        "data/",
        "data/audio/audio/",
        train=True,
        transform=nn.Sequential(AudioAug(0.25), LogMelSpectrogram()),
    )
    test_dataset = AudioDataset(
        "data/",
        "data/audio/audio/",
        train=False,
        transform=LogMelSpectrogram(),
    )

    batch_size = 8

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = nn.Sequential(
        nn.Conv2d(1, 8, 3),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.AvgPool2d(2),  # 64 x 215
        nn.Conv2d(8, 16, 3),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AvgPool2d(2),  # 32 x 107
        nn.Conv2d(16, 32, 3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AvgPool2d(2),  # 16 x 53
        nn.Conv2d(32, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AvgPool2d(2),  # 8 x 26
        nn.Conv2d(64, 128, 3),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 50),
    )
    model.to(device)

    training(model, device, train_loader, test_loader)


if __name__ == "__main__":
    main()
