import numpy as np
from scipy.io import wavfile
from scipy import signal

from python_speech_features import mfcc

L = 16000


def read_file(filepath):
    _, data = wavfile.read(filepath)
    return adapt(data)


def get_spectrogram(data):
    f, t, arr = signal.spectrogram(data)
    return arr


def parse_file(filepath):
    data = read_file(filepath)
    spectrogram = get_spectrogram(data)
    features = mfcc(data)
    return data, spectrogram, features


def adapt(audio):
    if len(audio) == L:
        return audio
    elif len(audio) < L:
        return np.pad(audio, pad_width=int((L - len(audio)) / 2), mode='constant', constant_values=0)
    else:
        start = np.random.randint(0, len(audio) - L)
        return audio[start: start + L]
