import tensorflow as tf
sr = fs = 24000
import pyworld
import matplotlib.pyplot as plt
import simpleaudio
import os
import librosa
from scipy.io import wavfile
import numpy as np
import noisereduce
import pyworld
from utilities import wave_play, visualize

def load_models(save_dir, f0_model_name, mel_model_name, mel2sp_model_name):
    f0_model = tf.keras.models.load_model(os.path.join(save_dir, f0_model_name))
    mel_model = tf.keras.models.load_model(os.path.join(save_dir, mel_model_name))
    sp2mel_model = tf.keras.models.load_model(os.path.join(save_dir, mel2sp_model_name))
    return f0_model, mel_model, sp2mel_model

def wav_file_loader(wav_path):
    wav =  librosa.load(wav_path, sr=sr)[0]
    wav = wav.astype(np.float64)
    return wav

def wav_file_writer(wav, wav_path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(wav_path, sr, wav.astype(np.int16))

def noise_reducer(wav, noise_time=0.1):
    noise = wav[:int(sr * noise_time)]
    noise_reduced_wav = noisereduce.reduce_noise(wav, noise)
    return noise_reduced_wav

def no_sound_interval(interval, len_wav):
    non_interval = list()
    for i in range(len(interval)-1):
        cur_inv = interval[i]
        nxt_inv = interval[i+1]
        if i ==1 and cur_inv[0] != 0:
            non_interval.append((0, cur_inv[0]))
        if i==len(interval) -2 and nxt_inv[1] != len_wav:
            non_interval.append((nxt_inv[1], len_wav))
        non_interval.append((cur_inv[1], nxt_inv[0]))
    return non_interval

def no_sound_trimming(wav, top_db=40):
    intervals = librosa.effects.split(wav, top_db=top_db, ref=np.mean, frame_length=256, hop_length=256)
    non_intervals = no_sound_interval(intervals, wav.shape[0])
    res_wav = librosa.effects.remix(wav, intervals=intervals)
    return res_wav, non_intervals

def wav_splitting(wav):
    f0, time_axis = pyworld.harvest(wav, fs)
    sp = pyworld.cheaptrick(wav, f0, time_axis, fs)
    ap = pyworld.d4c(wav, f0, time_axis, fs)
    f0 = (f0 / 800.0)
    sp = normalize2(sp)
    ap = normalize2(ap)
    return f0, sp, ap

def splitter(f0, sp, ap,  base_length=128):
    f0_dataset = list()
    sp_dataset = list()
    length = (len(f0) // base_length + 1) * base_length
    f0 = data_padding(f0, length, 'f0')
    sp = data_padding(sp, length, 'sp')
    ap = data_padding(ap, length, 'ap')
    for i in range(len(f0) // base_length):
        f0_clip = f0[i * base_length: (i + 1) * base_length]
        sp_clip = sp[i * base_length: (i + 1) * base_length]
        f0_dataset.append(f0_clip.astype(np.float32))
        sp_dataset.append(sp_clip.astype(np.float32))
    return f0_dataset, sp_dataset, ap

def generation(model, dataset, mode='sp'):
    res_data = list()
    for data in dataset:
        data = np.expand_dims(data, axis=-1)
        data = np.expand_dims(data, axis=0)
        data = model(data)
        data = np.squeeze(data, axis=-1)
        data = np.squeeze(data, axis=0)
        res_data.append(data)
    res_data = np.concatenate(res_data, axis=0)
    res_data = res_data.astype(np.float64)
    if mode=='f0':
        res_data = (res_data  * 800.0)
    return res_data

def synthesize(f0, sp, ap, wav_playing=False):
    wav = pyworld.synthesize(f0, sp, ap, fs)
    if wav_playing:
        wave_play(wav, fs)
    return wav

def data_padding(data, length, mode):
    if mode == 'f0':
        data = np.pad(data, [0, length - len(data)], 'constant')
    elif mode in ['sp', 'ap']:
        data = np.pad(data, ((0, length - len(data)), (0, 0)), 'constant')
    else:
        raise ValueError('Data padding mode must be f0, sp or ap')
    return data

def normalize2(wav):
    return (wav - np.min(wav)) / (np.max(wav) - np.min(wav))

def converter():
    #wavの読み込み
    #ノイズの消去
    #無音領域のトリミング >> Triming Position を保存しておく
    #データの正規化
    #データのカッティング >> padding もついでに
    #Data の Generation
    #Padding の場合、最後の部分の消去
    #無音領域の再構成
    #データをつなげる
    wav_path = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/ProductionModel/DataStore2/Test.wav'
    model_path = './ModelData2'
    model_name = 'pix2pix_127_freq_generator'
    f0_model_name = model_name + '_f0_B8_ADV0_sigmoid'
    sp_model_name = model_name + '_epoch_270_sp'
    wav = wav_file_loader(wav_path)
    f0_model, sp_model = load_models(model_path, f0_model_name, sp_model_name)
    wav = noise_reducer(wav)
    wav, non_intervals = no_sound_trimming(wav)
    wave_play(wav)
    f0, sp, ap = wav_splitting(wav)
    visualize(f0)
    f0_dataset, sp_dataset, ap = splitter(f0, sp, ap)
    f0 = generation(f0_model, f0_dataset, mode='f0')
    # f0 = np.concatenate(f0_dataset, axis=0).astype(np.float64) + 400.0 * 400 + 100
    sp = generation(sp_model, sp_dataset)
    visualize(sp)
    # sp = np.concatenate(sp_dataset, axis=0).astype(np.float64)
    visualize(f0)
    wav = synthesize(f0, sp, ap, wav_playing=True)

if __name__=='__main__':
    converter()