from utils import *
import simpleaudio
import pyworld
from pprint import pprint
import numpy as np
import librosa
import tensorflow as tf
from dtwalign import dtw
from fastdtw import fastdtw
import pysptk


#CONSTANTS
wave_path = 'TargetVoices/jvs093/parallel100/wav24kHz16bit/VOICEACTRESS100_025.wav'
wave_dir = 'TargetVoices/jvs093/parallel100/wav24kHz16bit/'
fs = sr = 24000
src_path = 'SourceVoices/jvs034/parallel100/wav24kHz16bit/VOICEACTRESS100_025.wav'
# src_path = '../test-voices/test1.wav'
dst_path = 'TargetVoices/jvs093/parallel100/wav24kHz16bit/VOICEACTRESS100_025.wav'

num_freq = 1025
sample_rate = 24000
frame_length_ms = 50
frame_shift_ms=12.5

def wave_play(wav):
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

def load_wave(wave_path, sr=fs):
    wav, _ = librosa.load(wave_path, sr=sr, mono=True)
    return wav

def DDTW(S):
    S_prime = S.copy()
    for i in range(1, len(S)-1):
        S_prime[i] = (S[i]- 1.5 *S[i-1]+ 0.5*S[i+1])/2
    return S_prime

def dtwTest(src_path, dst_path):
    src = load_wave(src_path)
    dst = load_wave(dst_path)
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    #このdst_f0に合わせて他のを整列させる？
    src_f0, src_t = pyworld.harvest(src, fs=fs)
    dst_f0, dst_t = pyworld.harvest(dst, fs=fs)
    src_sp = pyworld.cheaptrick(src, src_f0, src_t, fs=fs)
    dst_sp = pyworld.cheaptrick(dst, dst_f0, dst_t, fs=fs)
    src_ap = pyworld.d4c(src, src_f0, src_t, fs=fs)
    dst_ap = pyworld.d4c(dst, dst_f0, dst_t, fs=fs)
    src_mc = pysptk.sp2mc(src_sp, order=10, alpha=pysptk.util.mcepalpha(fs))
    dst_mc = pysptk.sp2mc(dst_sp, order=10, alpha=pysptk.util.mcepalpha(fs))
    # visualize(src_f0)
    # visualize(dst_f0)
    # visualize(src_ap)
    # visualize(dst_ap)
    # visualize(src_sp)
    # visualize(dst_sp)
    # f0res = dtw(src_f0, dst_f0)
    # f0res = dtw(src_ap, dst_ap)
    # visualize(src_mc)
    # visualize(dst_mc)

    # f0res = dtw(DDTW(src_mc), DDTW(dst_mc))
    f0res = dtw(src_mc, dst_mc)
    # print(f0res.shape)
    # syn_src_wav = src[f0res.path[:, 0]]
    # syn_dst_wav = dst[f0res.path[:, 1]]
    # src_stft = stft(src)
    # dst_stft = stft(dst)
    # src_stft_aligned = src_stft[f0res.path[:, 0], :]
    # dst_stft_aligned = dst_stft[f0res.path[:, 1], :]
    # syn_src_wav = invers_stft(src_stft_aligned)
    # syn_dst_wav = invers_stft(dst_stft_aligned)
    # dist, path = fastdtw(src_f0, dst_f0)
    #f0res >> (1827, 2)
    src_f0_aligned = src_f0[f0res.path[:, 0]]
    dst_f0_aligned = dst_f0[f0res.path[:, 1]]
    src_ap_aligned = src_ap[f0res.path[:, 0], :]
    dst_ap_aligned = dst_ap[f0res.path[:, 1], :]
    src_sp_aligned = src_sp[f0res.path[:, 0], :]
    dst_sp_aligned = dst_sp[f0res.path[:, 1], :]
    # visualize(src_f0_aligned)
    # visualize(dst_f0_aligned)
    # visualize(src_ap_aligned)
    # visualize(dst_ap_aligned)
    # visualize(src_sp_aligned)
    # visualize(dst_sp_aligned)
    syn_src_wav = pyworld.synthesize(src_f0_aligned, src_sp_aligned, src_ap_aligned, fs=fs)
    syn_dst_wav = pyworld.synthesize(dst_f0_aligned, dst_sp_aligned, dst_ap_aligned, fs=fs)
    wave_play(syn_src_wav)
    wave_play(syn_dst_wav)

def stft(y):
    n_fft = (num_freq - 1) * 2
    hop_length = int(frame_shift_ms / 1000 * sample_rate)
    win_length = int(frame_length_ms / 1000 * sample_rate)
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def invers_stft(y):
    hop_length = int(frame_shift_ms / 1000 * sample_rate)
    win_length = int(frame_length_ms / 1000 * sample_rate)
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)

def mel(y):
    n_fft = (num_freq - 1) * 2
    hop_length = int(frame_shift_ms / 1000 * sample_rate)
    win_length = int(frame_length_ms / 1000 * sample_rate)
    return librosa.feature.melspectrogram(y, sr=sr, hop_length=hop_length, win_length=win_length, n_fft=n_fft)

def mfcc(y):
    mel_spec = mel(y)
    return librosa.feature.mfcc(S=librosa.power_to_db(mel_spec))

if __name__=='__main__':
    dtwTest(src_path, dst_path)
