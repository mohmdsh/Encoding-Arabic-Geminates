import numpy as np
import librosa
from pathlib import Path
from config import TRAD_SR, TRAD_N_FFT, TRAD_HOP_LENGTH


def get_mfcc(wav_path: str, n_mfcc: int = 13) -> np.ndarray:
    y, sr = librosa.load(wav_path, sr=TRAD_SR)
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=TRAD_N_FFT, hop_length=TRAD_HOP_LENGTH,
    ).T


def get_fbank(wav_path: str, n_mels: int = 80) -> np.ndarray:
    y, sr = librosa.load(wav_path, sr=TRAD_SR)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        n_fft=TRAD_N_FFT, hop_length=TRAD_HOP_LENGTH,
    )
    return librosa.power_to_db(S, ref=np.max).T


def get_melspec(wav_path: str, n_mels: int = 128) -> np.ndarray:
    y, sr = librosa.load(wav_path, sr=TRAD_SR)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        n_fft=TRAD_N_FFT, hop_length=TRAD_HOP_LENGTH,
    )
    return librosa.power_to_db(S, ref=np.max).T


TRAD_FEATS = {
    "MFCC-13":     lambda p, fr, to: get_mfcc(p, n_mfcc=13)[fr:to].mean(axis=0),
    "MFCC-40":     lambda p, fr, to: get_mfcc(p, n_mfcc=40)[fr:to].mean(axis=0),
    "Fbank-80":    lambda p, fr, to: get_fbank(p, n_mels=80)[fr:to].mean(axis=0),
    "MelSpec-128": lambda p, fr, to: get_melspec(p, n_mels=128)[fr:to].mean(axis=0),
}
