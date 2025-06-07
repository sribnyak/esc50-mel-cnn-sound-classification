"""Module with data transforms."""

import random

import numpy as np
import torch
import torch.nn as nn
import librosa


# TODO refactor functions


def compute_log_mel_spect(audio, sample_rate):
    # # Size of the Fast Fourier Transform (FFT),
    # # which will also be used as the window length
    # n_fft = 1024
    # hop_length = 512
    # mel_bins = 60
    Mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_spectrogram_db = librosa.power_to_db(Mel_spectrogram)
    return mel_spectrogram_db


class LogMelSpectrogram(nn.Module):
    """Module that computes the log-mel spectrogram of an audio signal.

    For details see compute_log_mel_spect, this is just a nn.Module wrapper."""

    def __init__(self):
        """Initialize the module."""
        super().__init__()

    def forward(
        self, x: tuple[torch.Tensor, int, int]
    ) -> tuple[torch.Tensor, int]:
        """Compute the log-mel spectrogram of an audio signal.

        Args:
            x (tuple[torch.Tensor, int, int]): audio data, sample rate, label.

        Returns:
            tuple[torch.Tensor, int]: log-mel spectrogram, label.
        """
        audio, sr, target = x
        log_mel = compute_log_mel_spect(audio.numpy(), sr)
        return torch.from_numpy(log_mel.astype(np.float32)), target


def add_noise(audio, noise_factor=0.1):
    noise = np.random.normal(0, noise_factor, len(audio))
    return audio + noise


def pitch_shifting(audio, sr, strength=2):
    pitch_change = strength * 2 * (np.random.uniform())
    audio = librosa.effects.pitch_shift(
        y=audio, sr=sr, n_steps=pitch_change, bins_per_octave=12
    )
    return audio


def random_shift(data):
    timeshift_fac = (
        0.2 * 2 * (np.random.uniform() - 0.5)
    )  # up to 20% of length
    length = data.shape[0]
    start = int(length * timeshift_fac)
    if start > 0:
        data = np.pad(data, (start, 0), mode="constant")[:length]
    else:
        data = np.pad(data, (0, -start), mode="constant")[:length]
    return data


def volume_scaling(data):
    dyn_change = np.random.uniform(low=1.5, high=2.5)
    data = data * dyn_change
    return data


def time_stretching(data, rate=1.5):
    input_length = len(data)
    streching = data.copy()
    streching = librosa.effects.time_stretch(y=streching, rate=rate)

    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(
            data, (0, max(0, input_length - len(streching))), "constant"
        )
    return streching


def random_augment(audio, sr):
    audio = add_noise(audio, noise_factor=0.1)
    audio = pitch_shifting(audio, sr, strength=2)
    audio = random_shift(audio)
    audio = volume_scaling(audio)
    strech_rate = np.random.normal(1, 0.2)
    audio = time_stretching(audio, rate=strech_rate)
    return audio


class AudioAug(nn.Module):
    """Augmentation transform for audio data.

    Applies random_augment function with probability aug_rate. For details
        of the augmentation see random_augment."""

    def __init__(self, aug_rate: float = 0.75):
        """Initialize the augmentation transform.

        Args:
            aug_rate (float, optional): probability of applying augmentation
                transform. Defaults to 0.75.
        """
        super().__init__()
        self.aug_rate = aug_rate

    def forward(
        self, x: tuple[torch.Tensor, int, int]
    ) -> tuple[torch.Tensor, int, int]:
        """Apply the augmentation transform with probability aug_rate.

        Args:
            x (tuple[torch.Tensor, int, int]): audio data, sample rate, label.

        Returns:
            tuple[torch.Tensor, int, int]: augmented audio data, same
                sample rate, same label.
        """
        if random.random() < self.aug_rate:  # TODO seed
            audio, sr, target = x
            aug = random_augment(audio.numpy(), sr)
            return (torch.from_numpy(aug.astype(np.float32)), sr, target)
        return x
