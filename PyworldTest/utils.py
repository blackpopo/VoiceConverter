import os
import librosa
from tqdm import trange
import numpy as np
import pyworld
import matplotlib.pyplot as plt
import simpleaudio


def load_wavs(wav_dir, sr=24000):
    #使用librosa读取音频
    wavs = dict()
    ori_wav = os.listdir(wav_dir)
    for i in trange(len(ori_wav)):
        file = ori_wav[i]
        file_path = os.path.join(wav_dir, file)
        wav, _ = librosa.load(file_path, sr = sr, mono = True)
        #wav = wav.astype(np.float64)
        wavs[file] = wav
    return wavs

def load_wave(wave_path, sr=24000):
    wav, _ = librosa.load(wave_path, sr=sr, mono=True)
    return wav

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

def world_decompose(wav, fs, frame_period = 5.0):

    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
#    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 20.0, f0_ceil = 3500.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    print("f0: {}, timeaxis:{}, sp:{}, ap{}")
    return f0, timeaxis, sp, ap

def world_encode_data(wavs, fs, frame_period = 5.0, coded_dim = 24):

    f0s = []
    timeaxes = []
    sps = []
    aps = []
    coded_sps = []

    for i in trange(len(wavs)):
        wav = wavs[i]
        f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = fs, frame_period = frame_period)
        coded_sp = world_encode_spectral_envelop(sp = sp, fs = fs, dim = coded_dim)
        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        coded_sps.append(coded_sp)

    return f0s, timeaxes, sps, aps, coded_sps

def world_encode_spectral_envelop(sp, fs, dim = 24):

    # Get Mel-cepstral coefficients (MCEPs)

    #sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)

    return coded_sp

def wav_padding(wav, sr, frame_period, multiple = 4):

    assert wav.ndim == 1
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values = 0)
    print(num_frames, num_frames_padded)
    print(num_frames_diff)
    print(num_pad_left, num_pad_right)
    return wav_padded

def visualize(array):
    if(array.ndim==1):
        plt.plot(array)
    else:
        plt.plot(array)
    plt.show()

def create_col(wav, sr):
    # Generation of Mel Filter Banks
    mel_basis = librosa.feature.melspectrogram(wav, sr)
    assert mel_basis.shape == (30, 513)
    logmel = librosa.core.power_to_db(mel_basis)
    return logmel