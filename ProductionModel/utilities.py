import random
import numpy as np
from librosa import display

class Dataset:
    def __init__(self, source_data, target_data):
        self.source_data = np.array(source_data)
        self.target_data = np.array(target_data)
        assert self.source_data.shape[0] == self.target_data.shape[0]
        self.length = self.source_data.shape[0]

    def shuffle(self):
        index = list(range(self.source_data.shape[0]))
        random.shuffle(index)
        temp_src = [self.source_data[i] for i in index]
        temp_tgt = [self.target_data[i] for i in index]
        self.source_data = np.array(temp_src)
        self.target_data = np.array(temp_tgt)

    def batch(self, batch_size):
        iter =   self.length // batch_size
        temp_source, temp_target = list(), list()
        for i in range(iter):
            temp_source.append(self.source_data[i * batch_size: (i+1)*batch_size])
            temp_target.append(self.target_data[i * batch_size: (i+1) * batch_size])
        if self.length % batch_size != 0:
            temp_source.append(self.source_data[self.length//batch_size * batch_size:])
            temp_target.append(self.target_data[self.length // batch_size * batch_size:])
        self.source_data = np.array(temp_source)
        self.target_data = np.array(temp_target)

    def get(self):
        return [self.source_data, self.target_data]

import tensorflow as tf
fs = 24000
import pyworld
import matplotlib.pyplot as plt
import simpleaudio
import os

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


def wav_padding(wav, sr, frame_period, multiple = 4):
    assert wav.ndim == 1
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values = 0)
    return wav_padded

def data_padding(data, length, mode):
    if mode == 'f0':
        data = np.pad(data, [0, length - len(data)], 'constant')
    else:
        data = np.pad(data, ((0, length - len(data)), (0, 0)), 'constant', constant_values=1e-6)
    return data

def world_decompose(wav, fs, frame_period = 5.0):
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    #声の聞こえる範囲だけど、雑音はいるから却下
#     f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 20.0, f0_ceil = 3500.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    return f0, timeaxis, sp, ap


def visualize(data, title1=None, title2=None):
    try:
        plt.plot(data)
        if title1 is not None: plt.title(title1)
        plt.show()
    except:
        pass
    try:
        display.specshow(np.array(data, dtype=np.float32))
        if title2 is not None: plt.title(title2)
        plt.show()
    except:
        pass

def generate_images(model, test_input):
  prediction = model(test_input, training=True)
  for data in prediction:
      data = tf.squeeze(data, axis=2)
      visualize(data)

def log_normalize(x):
  x = x / np.max(np.abs(x))
  x = -np.log10(x)
  return x / np.max(x)

def rev_log_normalize(x, base=10):
  x = -(x * base)  # 10で声の大きさが変わる 093の場合は13が最適
  return 10 ** x

def normalize(wav):
    return wav/np.max(abs(wav))

def normalize2(wav):
    return (wav-np.min(wav))/(np.max(wav) - np.min(wav))

def load_npz(save_folder_path, file_name):
    path = os.path.join(save_folder_path, file_name + '.npz')
    return np.load(path)

