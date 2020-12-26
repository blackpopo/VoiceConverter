import noisereduce
import librosa
sr = 24000
noise_time = 0.1
import os
import numpy as np
from scipy.io import wavfile
import tqdm

def wav_file_reader(wav_path):
    return librosa.load(wav_path, sr=sr)[0]

def wav_file_writer(wav, wav_path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(wav_path, sr, wav.astype(np.int16))

def load_files(source_folder):
    files = os.listdir(source_folder)
    for file in tqdm.tqdm(files):
        wav_path = os.path.join(source_folder, file)
        wav = wav_file_reader(wav_path)
        wav = wav.astype(np.float64)
        noise = wav[:int(sr * noise_time)]
        noise_reduced_wav = noisereduce.reduce_noise(wav, noise)
        wav_path = os.path.join(source_folder, file.split('.')[0] + '.wav')
        wav_file_writer(noise_reduced_wav, wav_path)

def main():
    source_folders = [
        'C:/Users/Atsuya/Documents/SoundRecording/parallel100/127_parallel100',
        'C:/Users/Atsuya/Documents/SoundRecording/nonpara30/084/127_nonpara30',
        'C:/Users/Atsuya/Documents/SoundRecording/nonpara30/093/127_nonpara30',
    ]
    for folder in source_folders:
        load_files(folder)

if __name__=='__main__':
    main()