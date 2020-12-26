import numpy as np
import pyworld
import matplotlib.pyplot as plt
from scipy.io import wavfile
fs = 24000
import simpleaudio
# ap はcode すれば3次元でいいらしい？
import  os
from tqdm import trange
import tensorflow as tf
from ast import literal_eval
import librosa
import librosa.display
import pysptk


def visualize(data):
    plt.plot(data)
    plt.show()

def image_visualize(data):
    plt.imshow(data, cmap='Greys_r')

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

def normalize(data):
    return data / np.max(np.abs(data))

def one_hot_test():
    # wave_play(wav, fs=24000)
    f0, timeaxis = pyworld.harvest(wav ,fs)
    sp_513 = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap_513 = pyworld.d4c(wav, f0, timeaxis, fs)
    sp_513 = normalize(sp_513)
    ap_513 = normalize(ap_513)
    # sp_513 = np.expand_dims(sp_513, -1)
    # sp_513 = pyworld.code_spectral_envelope(sp_513, fs, 32)
    # sp_513 = pysptk.sp2mc(sp_513, order=32, alpha=0.49)
    # print(sp_513.shape)
    librosa.display.specshow(sp_513[:256])
    plt.show()
    # segmap = tf.one_hot(sp_513, depth=1)
    # segmap = tf.nn.max_pool(sp_513, [1, sp_513.shape[1], sp_513.shape[2], 1], [1, 1, 1, 1], padding='SAME')

    # segmap_onehot, label_map, segmap_img = load_segmap(sp_513)
    sp_513 = one_hot(sp_513)

    # segmap = tf.squeeze(segmap, axis=3)
    # segmap = tf.squeeze(segmap, axis=0)
    # visualize(segmap)


def preprocess(x):
    color_value_dict = dict()

    print(x.shape)
    print("segmap_label no exists ! ")
    x_img_list = []
    label = 1

    h, w, c = x.shape

    x_img_list.append(x)

    for i in range(h) :
        for j in range(w) :
            if tuple(x[i, j, :]) not in color_value_dict.keys() :
                color_value_dict[tuple(x[i, j, :])] = label
    return color_value_dict


def load_segmap(segmap_img):

    color_value_dict = preprocess(segmap_img)

    segmap_img = np.expand_dims(segmap_img, axis=-1)

    label_map = convert_from_color_segmentation(color_value_dict, segmap_img)

    segmap_onehot = get_one_hot(label_map, len(color_value_dict))

    segmap_onehot = np.expand_dims(segmap_onehot, axis=0)

    """
    segmap_x = tf.read_file(image_path)
    segmap_decode = tf.image.decode_jpeg(segmap_x, channels=img_channel, dct_method='INTEGER_ACCURATE')
    segmap_img = tf.image.resize_images(segmap_decode, [img_height, img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label_map = convert_from_color_segmentation(color_value_dict, segmap_img, tensor_type=True)
    segmap_onehot = tf.one_hot(label_map, len(color_value_dict))
    segmap_onehot = tf.expand_dims(segmap_onehot, axis=0)
    """

    return segmap_onehot, label_map, segmap_img


def get_one_hot(targets, nb_classes):

    x = np.eye(nb_classes)[targets]

    return x

def convert_from_color_segmentation(color_value_dict, arr_3d):
    arr_2d = np.zeros((np.shape(arr_3d)[0], np.shape(arr_3d)[1]), dtype=np.uint8)

    for c, i in color_value_dict.items():
        color_array = np.asarray(c, np.float32).reshape([1, 1, -1])
        m = np.all(arr_3d == color_array, axis=-1)
        arr_2d[m] = i

    return arr_2d

def one_hot(sp):
    sp = tf.expand_dims(sp, -1)
    segmap = tf.zeros_like(sp)
    mean = tf.reduce_mean(sp)
    # for i in range(segmap.shape[-3]):
    #     for j in range(segmap.shape[-2]):
    #         if sp[i, j] > mean:
    #             segmap[i, j] = 1
    # segmap[tf.where(sp >= mean)] = 1
    # mean = np.mean(sp)
    #
    # mean = np.mean(sp)
    # segmap = np.zeros(sp.shape)
    # segmap[np.where(sp > mean)] = 1

    segmap = np.squeeze(segmap, -1)
    librosa.display.specshow(segmap)
    plt.show()
    # return sp_zeros

if __name__ == '__main__':
    wav_source = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/093_aligned_experiment/093_093_001.wav'
    _, wav = wavfile.read(wav_source)
    wav = wav.astype(np.float64)
    one_hot_test()