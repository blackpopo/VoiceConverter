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
    mel2sp_model = tf.keras.models.load_model(os.path.join(save_dir, mel2sp_model_name))
    return f0_model, mel_model, mel2sp_model

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

def wav_splitting(wav, hop_length_sec=0.005):
    f0, time_axis = pyworld.harvest(wav, fs)
    sp = pyworld.cheaptrick(wav, f0, time_axis, fs)
    ap = pyworld.d4c(wav, f0, time_axis, fs)
    f0 = (f0 / 800.0)
    sp = normalize2(sp)
    ap = normalize2(ap)
    mel = librosa.feature.melspectrogram(wav, n_fft=pyworld.get_cheaptrick_fft_size(fs, 71.0),
                                          hop_length=librosa.time_to_samples(hop_length_sec, sr), sr=sr).T
    mel = normalize2(mel)
    return f0, sp, ap, mel

def splitter(f0, sp, ap, mel,  base_length=128):
    f0_dataset = list()
    mel_dataset = list()
    length = (len(f0) // base_length + 1) * base_length
    f0 = data_padding(f0, length, 'f0')
    sp = data_padding(sp, length, 'sp')
    ap = data_padding(ap, length, 'ap')
    mel = data_padding(mel, length, 'mel')
    for i in range(len(f0) // base_length):
        f0_clip = f0[i * base_length: (i + 1) * base_length]
        mel_clip = mel[i * base_length: (i + 1) * base_length]
        f0_dataset.append(f0_clip.astype(np.float32))
        mel_dataset.append(mel_clip.astype(np.float32))
    return f0_dataset, sp, ap, mel_dataset

def f0_generation(f0_model, dataset):
    res_data = list()
    for data in dataset:
        data = np.expand_dims(data, axis=-1)
        data = np.expand_dims(data, axis=0)
        data = f0_model(data)
        data = np.squeeze(data, axis=-1)
        data = np.squeeze(data, axis=0)
        res_data.append(data)
    res_data = np.concatenate(res_data, axis=0)
    res_data = res_data.astype(np.float64)
    res_data = (res_data  * 800.0)
    return res_data

def mel2sp_generation(dataset, mel2sp_model, mel_model=None):
    res_data = list()
    for data in dataset:
        data = np.expand_dims(data, axis=-1)
        data = np.expand_dims(data, axis=0)
        if mel_model is not None:
            data = mel_model(data)
        data = mel2sp_model(data)
        data = np.squeeze(data, axis=-1)
        data = np.squeeze(data, axis=0)
        res_data.append(data)
    res_data = np.concatenate(res_data, axis=0)
    res_data = res_data.astype(np.float64)
    return res_data

def mel_generation(mel_model, dataset):
    res_data = list()
    for data in dataset:
        visualize(data, title1='Pre', title2='Pre')
        data = np.expand_dims(data, axis=-1)
        data = np.expand_dims(data, axis=0)
        data = mel_model(data)
        data = np.squeeze(data, axis=-1)
        data = np.squeeze(data, axis=0)
        visualize(data, title2='Post', title1='post')
        res_data.append(data)
    res_data = np.concatenate(res_data, axis=0)
    res_data = res_data.astype(np.float64)
    return res_data

def synthesize(f0, sp, ap, wav_playing=False):
    wav = pyworld.synthesize(f0, sp, ap, fs)
    if wav_playing:
        wave_play(wav, fs)
    return wav

def data_padding(data, length, mode):
    if mode == 'f0':
        data = np.pad(data, [0, length - len(data)], 'constant')
    elif mode in ['sp', 'ap', 'mel', 'mel2sp']:
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
    # f0_model_name = model_name + '_f0_B8_ADV0_sigmoid'
    # mel_model_name = model_name + '_1D_epoch_80_mel'
    # mel_model_name = model_name +'_epoch_20_mel'
    mel2sp_model_name = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/ProductionModel/SRGAN/ModelData2/srgan_generator_epoch_30_mel2sp'
    # wav = wav_file_loader(wav_path)
    # f0_model, mel_model, mel2sp_model = load_models(model_path, f0_model_name, mel_model_name, mel2sp_model_name)
    # wav = noise_reducer(wav)
    # wav, non_intervals = no_sound_trimming(wav)
    # wave_play(wav)
    # f0, sp, ap, mel = wav_splitting(wav)
    # visualize(mel)
    # f0_dataset, sp, ap, mel_dataset = splitter(f0, sp, ap, mel)
    # # f0 = f0_generation(f0_model, f0_dataset)
    # f0 = np.concatenate(f0_dataset, axis=0).astype(np.float64) * 800 + 100
    # sp = mel2sp_generation(mel_model, mel2sp_model, mel_dataset)
    # # visualize(sp)
    # # sp = np.concatenate(sp_dataset, axis=0).astype(np.float64)
    # wav = synthesize(f0, sp, ap, wav_playing=True)
    # # mel = mel_generation(mel_model, mel_dataset)
    # # visualize(mel)
    # # mel = np.concatenate(mel_dataset, axis=0).astype(np.float64)
    # # wav = librosa.feature.inverse.mel_to_audio(mel.T, n_fft=pyworld.get_cheaptrick_fft_size(fs, 71.0),
    # #                                       hop_length=librosa.time_to_samples(0.005, sr), sr=sr)
    # print(wav.shape)
    # # wave_play(wav, fs)
    #
    _093_wav_path = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/jvs_ver1/jvs093/parallel100/wav24kHz16bit/VOICEACTRESS100_090.wav'
    _093_wav = wav_file_loader(_093_wav_path)
    visualize(_093_wav)
    # f0_model, mel_model, mel2sp_model = load_models(model_path, f0_model_name, mel_model_name, mel2sp_model_name)
    mel2sp_model = tf.keras.models.load_model(mel2sp_model_name)
    print(mel2sp_model)
    _093_wav = noise_reducer(_093_wav)
    _093_wav, non_intervals = no_sound_trimming(_093_wav)
    # wave_play(_093_wav)
    f0, sp, ap, mel = wav_splitting(_093_wav)
    f0_dataset, sp, ap, mel_dataset = splitter(f0, sp, ap, mel)
    f0 = np.concatenate(f0_dataset, axis=0).astype(np.float64) * 800 #f0問題ない

    sp_gen = mel2sp_generation(mel_dataset, mel2sp_model, None)
    visualize(sp_gen) #なぜか、sp_genだと上手く再生されない
    # visualize(sp)
    sp_gen = sp_gen + 1e-10
    wav = pyworld.synthesize(f0, sp_gen, ap, fs)
    visualize(wav)
    # wav = synthesize(f0, sp_gen, ap, wav_playing=True)
    wave_play(wav,fs)


if __name__=='__main__':
    converter()