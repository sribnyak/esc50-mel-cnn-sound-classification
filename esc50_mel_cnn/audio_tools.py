import numpy as np
import librosa


def compute_log_mel_spect(
    audio, sample_rate
):  # Size of the Fast Fourier Transform (FFT), which will also be used as the window length
    # n_fft = 1024
    # hop_length = 512
    # mel_bins = 60
    Mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    # Mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate,hop_length=hop_length, win_length=n_fft, n_mels = mel_bins)
    mel_spectrogram_db = librosa.power_to_db(Mel_spectrogram)
    return mel_spectrogram_db


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
    start = int(data.shape[0] * timeshift_fac)
    if start > 0:
        data = np.pad(data, (start, 0), mode="constant")[0 : data.shape[0]]
    else:
        data = np.pad(data, (0, -start), mode="constant")[0 : data.shape[0]]
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
