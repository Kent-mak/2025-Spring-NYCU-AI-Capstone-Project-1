import os
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset
from torchaudio.transforms import Resample, MFCC
from collections import Counter
import random
from torchaudio.utils import download_asset


class WavDataset(Dataset):
    def __init__(self, file_list, label_list, sample_rate=44100, transform=None, noise=None, reverb=False):
        """
        Args:
            file_list (list): List of audio file paths.
            label_list (list): Corresponding class labels.
            sample_rate (int): Target sample rate for all audio files.
            transform (callable, optional): Optional transform to apply to waveform.
        """
        self.file_list = file_list
        self.label_list = label_list
        self.sample_rate = sample_rate
        self.transform = transform
        self.noise=noise
        self.reverb = reverb

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        label = self.label_list[idx]

        # Load audio file
        waveform, sr = torchaudio.load(filepath)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if self.noise is not None:
            max_start = self.noise.size(1) - sr
            start_sample = random.randint(0, max_start)
            # print("start_sample", start_sample)
            noise_segment = self.noise[:, start_sample:start_sample + sr]
            snr_dbs = torch.Tensor([1.0])
            waveform = F.add_noise(waveform, noise_segment, snr=snr_dbs)

        # Apply transform if provided
        if self.transform:
            waveform = self.transform(waveform)


        return waveform, label  # Return waveform tensor and class label


    
def load_files_from_subdirectories(root_dir):
    """
    Scans `root_dir` and assigns a class label to each subdirectory.
    
    Returns:
        file_paths (list): List of audio file paths.
        labels (list): Corresponding class labels as integers.
        class_mapping (dict): Maps class labels (int) to class names (str).
    """
    file_paths = []
    labels = []
    class_mapping = {}  # Maps class index to class name

    # Iterate through subdirectories
    for idx, class_name in enumerate(sorted(os.listdir(root_dir))):  
        class_path = os.path.join(root_dir, class_name)
        
        if os.path.isdir(class_path):  # Ensure it's a directory
            class_mapping[idx] = class_name  # Assign index to class
            
            # Collect all .wav files in the class subdirectory
            for file in os.listdir(class_path):
                if file.endswith('.wav'):
                    file_paths.append(os.path.join(class_path, file))
                    labels.append(idx)  # Assign corresponding class label

    return file_paths, labels, class_mapping


def get_dataset(directory, sr, transform=None, noise_add=False):
    

    all_files, all_labels, class_mapping = load_files_from_subdirectories(directory)
    print("Class Mapping:", class_mapping)

    noise = None
    if noise_add:
        noise, noise_sr = torchaudio.load('./Data/house-party-inside-voices-talking-10-62063.wav')
        if noise_sr != sr:
            resampler = Resample(orig_freq=noise_sr, new_freq=sr)
            noise = resampler(noise)
        
        if noise.shape[0] > 1:
            # print("noise shape", noise.shape)
            noise = noise.mean(dim=0, keepdim=True)

    dataset = WavDataset(all_files, all_labels, sr, transform=transform, noise=noise)
    return dataset



if __name__ == "__main__":
    # Define dataset parameters
    dataset_directory = r".\Data\processed\train"
    sample_rate = 44100  

    mfcc_transform = MFCC(
        sample_rate=44100,  # Match dataset sample rate
        n_mfcc=13,  # Number of MFCC coefficients
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
    )

    train_set, train_loader= get_dataset(dataset_directory, sample_rate, transform=mfcc_transform, to_np=True)

    train_class_counts = Counter()

    for input, labels in train_loader:  # Iterate over batches
        print(input.shape)
        train_class_counts.update(labels.tolist())

    print("Train Class Counts:", train_class_counts)
