import librosa
import numpy as np
import soundfile as sf
import os

class AudioPreprocessor:
    def __init__(self, target_sr=44100, duration=1.0):
        """
        Initializes the preprocessor.
        :param target_sr: Target sampling rate (default: 44100 Hz).
        :param duration: Target duration in seconds (default: 1 second).
        """
        self.target_sr = target_sr
        self.target_samples = int(target_sr * duration)
    
    def preprocess_audio(self, file_path):
        """
        Loads an audio file, resamples it, and ensures it is exactly 1 second long.
        :param file_path: Path to the audio file.
        :return: Processed waveform (numpy array) and sampling rate.
        """
        y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        
        # Trim or pad to ensure 1 second duration
        if len(y) > self.target_samples:
            y = y[:self.target_samples]  # Trim
        else:
            y = np.pad(y, (0, self.target_samples - len(y)))  # Pad with zeros
        
        return y, self.target_sr
    
    
    def process_directory(self, input_dir, output_dir):
        """
        Processes all .wav files in the input directory and saves them to the output directory.
        :param input_dir: Directory containing input .wav files.
        :param output_dir: Directory to save processed .wav files.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for file_name in os.listdir(input_dir):
            if file_name.endswith(".wav"):
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, file_name)

                y, sr = self.preprocess_audio(input_path)
                sf.write(output_path, y, sr)

                print(f"Processed and saved: {output_path}")

# Example usage
if __name__ == "__main__":
    preprocessor = AudioPreprocessor(target_sr=44100)
    
    # Load and preprocess an example audio file
    file_path = "./Data/processed/train/Ceramic/1.wav"  # Replace with actual file path
    y, sr = librosa.load(file_path, sr=preprocessor.target_sr, mono=True)
    # input_directory = "./Data/raw/Marble"
    # output_directory = "./Data/processed/Marble"
    # preprocessor.process_directory(input_directory, output_directory)

