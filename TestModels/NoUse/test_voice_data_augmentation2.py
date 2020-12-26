from scipy.fftpack import rfft, irfft
from test_utils import *
from scipy.io import wavfile
from scipy import signal
import numpy as np
# samplerate = 24000


def lowpass(wav, freq):
    yf = rfft(wav)
    yf[freq:] = 0.0
    wav = irfft(yf)
    return wav


def gaussean_noise(wav, noise):
    return wav * noise

def shift(wav, shift):
    if shift < 0:
        return leftshit(wav, shift)
    else:
        return rightshit(wav, shift)

#shift 300 ~ 100
def leftshit(wav, shift):
    yf = rfft(wav)
    yf = np.roll(yf, shift)
    yf[shift:] = 0.0
    wav = irfft(yf)
    return wav

#shift 100 ~ 300
def rightshit(wav, shift):
    yf = rfft(wav)
    yf = np.roll(yf, shift)
    yf[:shift] = 0.0
    wav = irfft(yf)
    return wav


#p 広域を強調する強さ 1.2 ~ 0.8
def pre_emphasis(wave, p=0.5):
    # 係数 (1.0, -p) のFIRフィルタを作成
    return signal.lfilter([1.0, -p], 1, wave)

#信号を平準化します。 信号の高周波数成分を維持しつつ平準化したいときに効果的なフィルタ >> windowlength 10 ~ 100の奇数
def savgol_filter(wav, window_length):
    return signal.savgol_filter(wav, window_length, 1)

#10~50の奇数 >> 遅い…
def median_filter(wav, num):
    return signal.medfilt(wav, num)

#使える 0.9 ~ 1.1 * len(wav)
def resample_filter(wav, num):
    return signal.resample(wav, num)

def zero_factor(wav, zero_list):
    for i in zero_list:
        wav[i] = 0.0
    return wav

#こいつは必須！

#mode in 'lowpass', 'gaussean_noise', 'shift', 'pre_emphasis', 'savgol_filter', 'median_filter', 'resample_filter',  'zero_factor' of 8 specifications
def patch(src_wav, tgt_wav, mode=None):
    length = np.min((src_wav.shape[0], tgt_wav.shape[0]))
    if mode == 'lowpass':
        freq = np.random.randint(20000, 35000)
        rsrc = lowpass(src_wav, freq)
        rtgt = lowpass(tgt_wav, freq)
    elif mode == 'savgol_filter':
        window_length = np.random.randint(5, 44) * 2 + 1
        rsrc = lowpass(src_wav, window_length)
        rtgt = lowpass(tgt_wav, window_length)
    elif mode == 'gaussean_noise':
        noise = np.random.normal(loc=1, scale=0.2, size=length)
        rsrc = gaussean_noise(src_wav, noise)
        rtgt = gaussean_noise(tgt_wav, noise)
    elif mode == 'shift':
        shift_size = 0
        while (-100 < shift_size < 100):
            shift_size = np.random.randint(-300, 300)
        rsrc = shift(src_wav, shift_size)
        rtgt = shift(tgt_wav, shift_size)
    elif mode == 'pre_emphasis':
        p = np.random.normal(loc=0.97, scale=0.2)
        rsrc = pre_emphasis(src_wav, p)
        rtgt = pre_emphasis(tgt_wav, p)
    elif mode == 'median_filter':
        num = np.random.randint(5, 24) * 2 + 1
        rsrc = median_filter(src_wav, num)
        rtgt = median_filter(tgt_wav, num)
    elif mode == 'resample_filter':
        factor = np.random.rand() * 0.2 + 0.9
        rsrc = resample_filter(src_wav, int(length*factor))
        rtgt = resample_filter(tgt_wav, int(length*factor))
    elif mode == 'zero_factor':
        #1% to 5% makes loss
        zero_list = np.random.randint(0, length, int(length * (np.random.rand() * 0.04 + 0.01)))
        rsrc = zero_factor(src_wav, zero_list)
        rtgt = zero_factor(src_wav, zero_list)
    else:
        rsrc = src_wav
        rtgt = tgt_wav
    return rsrc, rtgt


def get_mode_names():
    return ['lowpass', 'gaussean_noise', 'shift', 'pre_emphasis', 'savgol_filter', 'median_filter', 'resample_filter',  'zero_factor']

def data_augmentations(wav_list):
    names = get_mode_names()
    res_dict = defaultdict()
    for name in names:
        res_dict[name] = list()
        print("{} is starting...".format(name))
        for i in trange(len(wav_list)):
            src_wav, tgt_wav = wav_list[i]
            rsrc_wav, rtgt_wav = patch(src_wav, tgt_wav, mode=name)
            res_dict[name].append((rsrc_wav, rtgt_wav))
    return res_dict

def data_augmentation(wav_list, name):
    res_list = list()
    for i in trange(len(wav_list)):
        src_wav, tgt_wav = wav_list[i]
        rsrc_wav, rtgt_wav = patch(src_wav, tgt_wav, mode=name)
        res_list.append((rsrc_wav, rtgt_wav))
    return res_list