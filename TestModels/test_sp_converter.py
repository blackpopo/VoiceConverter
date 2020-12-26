import tensorflow as tf
import pyworld
import librosa
import os
from tqdm import trange
import numpy as np
from test_utils import *
from test_loader import *

fs = sr = 24000

def load_wavs(wav_file_path):
    files = os.listdir(wav_file_path)
    res_wavs = list()
    for i in trange(len(files)):
        file = files[i]
        path = os.path.join(wav_file_path, file)
        wav = librosa.load(path, sr)[0]
        wav = wav.astype(np.float64)
        res_wavs.append(wav)
    return res_wavs

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def play_with_sp_converter():
    base_valid_path = '../test-voices/my_voices/'
    model_base = 'ModelData'

    wav_file_path = 'alignment'
    model_name = 'only_normalized_256_256'
    wavs = load_wavs(os.path.join(base_valid_path, wav_file_path))
    generator = load_model(os.path.join(model_base, model_name, 'generator_model'))
    for wav in wavs:
        f0, t, sp, ap = world_decompose(wav, fs)

        # sp_coded = pyworld.code_spectral_envelope(sp, fs ,256)
        ap = normalize(ap)
        sp = normalize(sp)
        visualize(sp)

        res_sps = list()
        wave_play(wav)

        for split_sp in split_factors(sp, length=256, mode='sp'):
            # split_sp = np.expand_dims(split_sp, 0) #if you use coded
            split_sp = np.expand_dims(split_sp[:, :256], 0) #else
            split_sp = np.expand_dims(split_sp, 3)
            res_sp = generator(split_sp)
            res_sp = np.squeeze(res_sp, axis=0)#Batch size, channelの削除
            res_sp = np.squeeze(res_sp, axis=2)
            res_sp = res_sp.astype(np.float64)
            # res_sp = rev_log_normalize(res_sp) #log normalized
            res_sps.append(res_sp) #(256, 256) >> (256, 513)に変換

        sp = np.concatenate(res_sps, axis=0)[:f0.shape[0]] #一次元方向にくっつける
        # sp = pyworld.decode_spectral_envelope(sp, fs, pyworld.get_cheaptrick_fft_size(fs)) # if you use code wav
        base_sp = np.full((f0.shape[0], 257), 1e-6)
        sp = np.concatenate([sp, base_sp], axis=1)
        visualize(sp)
        f0 += 300.0
        wav = pyworld.synthesize(f0, sp, ap, fs)
        wave_play(wav, fs)

if __name__=='__main__':
    play_with_sp_converter()



