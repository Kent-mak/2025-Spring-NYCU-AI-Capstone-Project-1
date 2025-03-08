import os
from pydub import AudioSegment

def convert_m4a_to_wav(directory):
    """
    Converts all .m4a files in the directory to .wav format.
    :param directory: Path to the folder containing .m4a files.
    """
    for filename in os.listdir(directory):
        if filename.lower().endswith(".m4a"):
            old_path = os.path.join(directory, filename)
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            wav_path = os.path.join(directory, wav_filename)

            # Convert .m4a to .wav
            audio = AudioSegment.from_file(old_path, format="m4a")
            audio.export(wav_path, format="wav")
            print(f"Converted: {filename} -> {wav_filename}")

def rename_files(directory):
    """
    Renames all files in the specified directory to integer indices while preserving extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()  # Sorting ensures consistency

    # Rename files with integer indices
    for index, filename in enumerate(files, start=1):
        old_path = os.path.join(directory, filename)
        _, ext = os.path.splitext(filename)  # Preserve the file extension
        new_filename = f"{index}{ext}"
        new_path = os.path.join(directory, new_filename)

        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path: ").strip()
    convert_m4a_to_wav(folder_path)
