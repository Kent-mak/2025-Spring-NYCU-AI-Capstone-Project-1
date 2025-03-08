
from torchaudio.transforms import MFCC
import numpy as np



mfcc_transform = MFCC(
    sample_rate=44100,  # Match dataset sample rate
    n_mfcc=13,  # Number of MFCC coefficients
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
)

def get_features_labels(dataset):

    features, labels = zip(*[dataset[i] for i in range(len(dataset))])
    features = np.stack(features)
    labels = np.array(labels)

    return features, labels
