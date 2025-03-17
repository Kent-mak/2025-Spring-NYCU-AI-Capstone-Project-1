import torchaudio
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt
import torch

def plot_waveform(wav_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(wav_path)
    
    # Convert to mono if it's stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Get time axis
    time_axis = torch.arange(waveform.shape[1]) / sample_rate
    
    # Plot the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis.numpy(), waveform.numpy().squeeze(), linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform of " + wav_path)
    plt.grid()
    plt.show()

# Example usage
wav_file = r".\Data\processed\train\Ceramic\15.wav"  # Replace with your .wav file path
plot_waveform(wav_file)
