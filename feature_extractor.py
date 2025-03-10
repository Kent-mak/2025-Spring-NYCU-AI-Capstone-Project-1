import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.decomposition import PCA




def mfcc_transform(waveform: torch.Tensor, sample_rate: int = 44100, n_mfcc: int = 13):
    """
    Computes MFCC features with mean pooling.

    Args:
        waveform (Tensor): Audio waveform of shape (1, T) or (C, T).
        sample_rate (int): Sample rate of the audio.
        n_mfcc (int): Number of MFCC coefficients.

    Returns:
        Tensor: MFCC features with shape (n_mfcc,).
    """
    # Ensure mono audio
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Define MFCC transform
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
    )

    # Compute MFCCs
    mfcc = mfcc_transform(waveform)  # Shape: (C, n_mfcc, time_frames)

    # Compute mean pooling across time dimension
    mfcc_mean = mfcc.mean(dim=2).squeeze(0)  # Shape: (n_mfcc,)
    return mfcc_mean
    # mfcc_var = mfcc.var(dim=2).squeeze(0)    # Shape: (n_mfcc,)
    # mfcc_features = torch.cat((mfcc_mean, mfcc_var), dim=0) 

    # return mfcc_features

class MelSpectrogramPCA:
    def __init__(self, sample_rate=44100, n_mels=128, n_fft=2048, hop_length=512, pca_components=20):
        """
        Args:
            sample_rate (int): Target sample rate.
            n_mels (int): Number of Mel frequency bins.
            n_fft (int): FFT window size.
            hop_length (int): Number of samples between successive frames.
            pca_components (int): Number of principal components to retain.
        """
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.pca = PCA(n_components=pca_components)

    def __call__(self, waveform: torch.Tensor):
    
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Compute Mel Spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to Log Scale (dB)
        mel_spec_db = self.amplitude_to_db(mel_spec).squeeze(0).numpy()  # Shape: (n_mels, time_frames)

        # Apply PCA (reducing time_frames dimension)
        mel_spec_db_T = mel_spec_db.T  # Shape: (time_frames, n_mels)
        mel_pca = self.pca.fit_transform(mel_spec_db_T)  # Shape: (time_frames, pca_components)

        # Compute mean and variance across time frames
        mel_pca_mean = torch.tensor(mel_pca.mean(axis=0), dtype=torch.float32)
        mel_pca_var = torch.tensor(mel_pca.var(axis=0), dtype=torch.float32)

        # Concatenate mean and variance
        mel_pca_features = torch.cat((mel_pca_mean, mel_pca_var), dim=0)  # Shape: (2 * pca_components,)

        return mel_pca_features
