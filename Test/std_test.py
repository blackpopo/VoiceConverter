import random
import numpy as np
from librosa import display
from scipy.io import wavfile
fs = 24000
import pyworld
import matplotlib.pyplot as plt
import simpleaudio
import os
import librosa
import pysptk
from nnmnkwii.preprocessing import trim_zeros_frames
from dtwalign import dtw

def librosa_remixing(wav, top_db):
    wav = wav.astype(np.float64)
    res_wav = librosa.effects.remix(wav, intervals=librosa.effects.split(wav, top_db=top_db, ref=np.mean, frame_length=256,  hop_length=256))
    return res_wav

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
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    #声の聞こえる範囲だけど、雑音はいるから却下
#     f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 20.0, f0_ceil = 3500.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    return f0, timeaxis, sp, ap


def visualize(data):
    plt.plot(data)
    plt.show()
    display.specshow(np.array(data, dtype=np.float32))
    plt.show()


def log_normalize(wav):
  wav = (wav-np.min(wav))/(np.max(wav) - np.min(wav))
  wav = -np.log10(wav)
  return wav / np.max(wav)

def rev_log_normalize(x, base=10):
  x = -(x * base)  # 10で声の大きさが変わる 093の場合は13が最適
  return 10 ** x

def normalize(wav):
    return wav/np.max(abs(wav))

def normalize2(wav):
    return (wav-np.min(wav))/(np.max(wav) - np.min(wav))

def read_wav(wav_path):
    _, wav = wavfile.read(wav_path)
    wav = wav.astype(np.float64)
    return wav


def src_alignment(src_wav, dst_wav):
    ############ 無音消去
    src = librosa_remixing(src_wav, top_db=50)
    dst = librosa_remixing(dst_wav, top_db=50)

    ############ 音声分解
    src_f0, src_t = pyworld.harvest(src, fs=fs)
    dst_f0, dst_t = pyworld.harvest(dst, fs=fs)
    src_sp = pyworld.cheaptrick(src, src_f0, src_t, fs=fs)
    dst_sp = pyworld.cheaptrick(dst, dst_f0, dst_t, fs=fs)
    src_ap = pyworld.d4c(src, src_f0, src_t, fs=fs)
    dst_ap = pyworld.d4c(dst, dst_f0, dst_t, fs=fs)

    visualize(src_sp)
    ########## 正規化 pattern1
    # src_f0 = (src_f0 - 400.0) / 400.0
    # dst_f0 = (dst_f0 - 400.0) / 400.0
    # src_sp = normalize(src_sp)
    # dst_sp = normalize(dst_sp)
    # src_ap = normalize(src_ap)
    # dst_ap = normalize(dst_ap)

    #先に正規化するとこいつがアウト！ >> 正規化は最後で！
    src_sp_coded = pysptk.sp2mc(src_sp, order=32, alpha=pysptk.util.mcepalpha(fs))
    src_sp_coded = trim_zeros_frames(src_sp_coded)
    dst_sp_coded = pysptk.sp2mc(src_sp, order=32, alpha=pysptk.util.mcepalpha(fs))
    dst_sp_coded = trim_zeros_frames(dst_sp_coded)
    #############アライメント
    try:
        res = dtw(src_sp_coded, dst_sp_coded, step_pattern="typeIb")
        path = res.get_warping_path(target='query')
        src_f0 = src_f0[path]
        src_sp = src_sp[path, :]
        src_ap = src_ap[path, :]
    except:
        pass

    ############ 正規化 pattern2
    # src_f0 = (src_f0 - 400.0) / 400.0
    # dst_f0 = (dst_f0 - 400.0) / 400.0
    src_sp = normalize2(src_sp)
    dst_sp = normalize2(dst_sp)
    src_ap = normalize2(src_ap)
    dst_ap = normalize2(dst_ap)


    return src_f0, dst_f0, src_sp, dst_sp, src_ap, dst_ap


if __name__=='__main__':
    src_wav = 'C:/Users/Atsuya/Documents/SoundRecording/nonpara30/093/127_nonpara30/127_024.wav'
    dst_wav = 'C:/Users/Atsuya/Documents/SoundRecording/nonpara30/093/093_nonpara30/093_024.wav'
    src_wav = read_wav(src_wav)
    dst_wav = read_wav(dst_wav)
    # f0, timeaxis, sp, ap = world_decompose(wav, fs, frame_period=5.0)
    # wav2 = pyworld.synthesize(f0, normalize2(sp), normalize2(ap), fs)
    # wave_play(wav2)
    src_f0, dst_f0, src_sp, dst_sp, src_ap, dst_ap = src_alignment(src_wav, dst_wav)
    wave_play(pyworld.synthesize(src_f0, src_sp, src_ap, fs))