import pyworld
fs = 24000
frame_period = 5
import librosa
import numpy as np
import simpleaudio
sr = 24000
import matplotlib.pyplot as plt
from librosa import display

def wave_play(wav, fs=24000):
    if type(wav) is str:
        wav_obj = simpleaudio.WaveObject.from_wave_file(wav)
        wav_obj.play().wait_done()
    elif type(wav) is np.ndarray:
        wav *= 32767 / max(abs(wav))
        audio = wav.astype(np.int16)
        play_obj = simpleaudio.play_buffer(audio, 1, 2, fs)
        play_obj.wait_done()
    else:
        raise ValueError("wav must be file path or ndarray!{}".format(type(wav)))

def syntest():
    wav = librosa.load('C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/my_voices/alignment/test6.wav' , sr=fs)[0]
    wav = wav.astype(np.float64)
    f0, t = pyworld.harvest(wav, fs)
    sp = pyworld.cheaptrick(wav,f0, t, fs)
    ap = pyworld.d4c(wav, f0, t, fs)
    sp[:, :256] = 1e-6
    # ap *= 10.0
    wav = pyworld.synthesize(f0, sp, ap, fs, frame_period=frame_period)
    wave_play(wav, fs)

def mel_test():
    wav = librosa.load('C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/my_voices/alignment/test5.wav' , sr=fs)[0]
    wav = wav.astype(np.float64)
    f0, t = pyworld.harvest(wav, fs)
    sp = pyworld.cheaptrick(wav,f0, t, fs)
    ap = pyworld.d4c(wav, f0, t, fs)
    print(sp.shape)
    print(pyworld.get_cheaptrick_fft_size(fs, 71.0))
    mel1 = librosa.feature.melspectrogram(wav, sr=sr, n_fft=1024, hop_length=librosa.time_to_samples(0.005,  sr)) #直にmelspcectrogram
    mel2 = librosa.feature.melspectrogram(S=sp.T, sr=sr, n_fft=1024, hop_length=librosa.time_to_samples(0.005,  sr))
    visualize(mel1)
    visualize(mel2)


def normalize2(wav):
    return ((wav-np.min(wav))/(np.max(wav) - np.min(wav)) -0.5 ) *2

def visualize(data, title1=None, title2=None):
    plt.plot(data)
    if title1 is not None: plt.title(title1)
    plt.show()
    display.specshow(np.array(data, dtype=np.float32))
    if title2 is not None: plt.title(title2)
    plt.show()

if __name__=='__main__':
    # syntest()
    mel_test()