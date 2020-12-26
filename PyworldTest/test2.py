from utils import *
import pyworld
from pprint import pprint
import numpy as np
import librosa
import tensorflow as tf
from dtwalign import dtw
from DTWtest import *
from TestTrimming import *

from DTW import DTW

# wave_path = '../CycleGanConverterExample/TargetVoices/parallel100/wav24kHz16bit/VOICEACTRESS100_025.wav'
wave_dir = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/CycleGanConverterExample/TargetVoices/jvs093/parallel100/wav24kHz16bit/'
fs = sr = 24000
src_path = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/CycleGanConverterExample/SourceVoices/jvs034/parallel100/wav24kHz16bit/VOICEACTRESS100_050.wav'
# src_path = '../test-voices/test1.wav'
dst_path = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/CycleGanConverterExample/TargetVoices/jvs093/parallel100/wav24kHz16bit/VOICEACTRESS100_050.wav'



# res.get_warping_path(target="query")
#
# WindowSize を小さくする
#
# librosa.feature.melspectrogram(y=y, sr=sr
#
# librosa.feature.mfcc(y=y, sr=sr)
#
# pyworld.code_spectral_envelope

def librosa_remixing(wav, top_db):
    wav = np.array(wav).astype(np.float32)
    res_wav = librosa.effects.remix(wav, intervals=librosa.effects.split(wav, top_db=top_db, ref=np.mean, frame_length=256,  hop_length=256))
    return res_wav

def test2(src_wav_path, dst_wav_path):
    src = load_wave(src_wav_path)
    dst = load_wave(dst_wav_path)
    wave_play(src)
    wave_play(dst)
    # visualize(src)
    # visualize(dst)
    # src = librosa_remixing(src, 20)
    # dst = librosa_remixing(dst, 20)
    src = librosa_remixing(src, top_db=50)
    dst = librosa_remixing(dst, top_db=50)
    window_size = abs(len(src) - len(dst)) * 2
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    # visualize(src)
    # visualize(dst)
    src_f0, src_t = pyworld.harvest(src, fs=fs)
    dst_f0, dst_t = pyworld.harvest(dst, fs=fs)
    #雑音とかも許す設定になっている
    # src_f0, src_t = pyworld.harvest(src, fs=fs, f0_floor = 20.0, f0_ceil = 3500.0)
    # dst_f0, dst_t = pyworld.harvest(dst, fs=fs, f0_floor = 20.0, f0_ceil = 3500.0)
    src_sp = pyworld.cheaptrick(src, src_f0, src_t, fs=fs)
    dst_sp = pyworld.cheaptrick(dst, dst_f0, dst_t, fs=fs)
    src_ap = pyworld.d4c(src, src_f0, src_t, fs=fs)
    dst_ap = pyworld.d4c(dst, dst_f0, dst_t, fs=fs)
    # src_sp_coded = pyworld.code_spectral_envelope(src_sp, fs, 32)
    # dst_sp_coded = pyworld.code_spectral_envelope(dst_sp, fs, 32)
    src_sp_coded = pysptk.sp2mc(src_sp, order=32, alpha=pysptk.util.mcepalpha(fs))
    dst_sp_coded = pysptk.sp2mc(dst_sp, order=32, alpha=pysptk.util.mcepalpha(fs))
    src_sp_coded  = trim_zeros_frames(src_sp_coded)
    dst_sp_coded = trim_zeros_frames(dst_sp_coded)
    # src_sp_coded = librosa.feature.mfcc(y=src, sr=sr)
    # dst_sp_coded = librosa.feature.mfcc(y=dst, sr=sr)
    # visualize(src_sp_coded)
    # visualize(dst_sp_coded)
    # src_sp_coded = mfcc(src)
    # dst_sp_coded = mfcc(dst)
    # src_sp_coded = ddtw(src_sp_coded)
    # dst_sp_coded = ddtw(dst_sp_coded)
    # visualize(src_sp)
    # visualize(dst_sp)
    # visualize(src_sp_coded)
    # visualize(dst_sp_coded)
    # dtw = DTW(src_sp_coded, dst_sp_coded)
    # src_sp_aligned = dtw.align(src_sp)
    # src_f0_aligned = dtw.align(src_f0)
    # src_ap_aligned = dtw.align(src_ap)
    # print(src_sp_aligned.shape)
    # print(src_ap_aligned.shape)
    # print(src_f0_aligned.shape)
    # print(dst_sp.shape)
    # print(dst_ap.shape)
    # print(dst_f0.shape)
    res = dtw(src_sp_coded, dst_sp_coded, step_pattern="typeIb")
    # visualize(src_sp_coded)
    # visualize(dst_sp_coded)
    path = res.get_warping_path(target='query')
    src_f0_aligned = src_f0[path]
    src_ap_aligned = src_ap[path, :]
    src_sp_aligned = src_sp[path, :]
    # dst_f0_aligned = dst_f0[path]
    # dst_ap_aligned = dst_ap[path, :]
    # dst_sp_aligned = dst_sp[path, :]
    # visualize(src_f0)
    visualize(src_f0_aligned)
    visualize(dst_f0)
    # visualize(dst_f0_aligned)
    # visualize(src_ap)
    # visualize(src_ap_aligned)
    # visualize(dst_ap)
    # visualize(dst_ap_aligned)
    # visualize(src_sp)
    visualize(src_sp_aligned)
    visualize(dst_sp)
    # visualize(dst_sp_aligned)
    # print(src_sp_aligned.shape)
    # print(src_ap_aligned.shape)
    # print(src_f0_aligned.shape)
    # print(dst_sp.shape)
    # print(dst_ap.shape)
    # print(dst_f0.shape)
    syn_src_wav = pyworld.synthesize(src_f0_aligned, src_sp_aligned, src_ap_aligned, fs=fs)
    syn_dst_wav = pyworld.synthesize(dst_f0, dst_sp, dst_ap, fs=fs)
    # visualize(syn_src_wav)
    # visualize(syn_dst_wav)
    # wave_play(src)
    wave_play(syn_src_wav)
    wave_play(syn_dst_wav)
    # src_mel = mel(src)
    # src_aligned_mel = mel(syn_src_wav)
    # dst_mel = mel(dst)

    # visualize(src_mel.T[::-1])
    # visualize(src_aligned_mel.T[::-1])
    # visualize(dst_mel.T[::-1])



def user_func(x, y):
    return  np.lpg(np.sqrt((x - y) ** 2))

def test3(src_wav_path, dst_wav_path):
    src = load_wave(src_wav_path)
    dst = load_wave(dst_wav_path)
    src = librosa_remixing(src, top_db=65)
    dst = librosa_remixing(dst, top_db=65)
    window_size = abs(len(src) - len(dst))
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    src_stft = stft(src)
    dst_stft = stft(dst)
    src_f0, src_t = pyworld.harvest(src, fs=fs)
    dst_f0, dst_t = pyworld.harvest(dst, fs=fs)
    src_sp = pyworld.cheaptrick(src, src_f0, src_t, fs=fs)
    dst_sp = pyworld.cheaptrick(dst, dst_f0, dst_t, fs=fs)
    # src_ap = pyworld.d4c(src, src_f0, src_t, fs=fs)
    # dst_ap = pyworld.d4c(dst, dst_f0, dst_t, fs=fs)
    # src_sp_coded = pyworld.code_spectral_envelope(src_sp, fs, 32)
    # dst_sp_coded = pyworld.code_spectral_envelope(dst_sp, fs, 32)
    src_sp_coded = pysptk.sp2mc(src_sp, order=64, alpha=pysptk.util.mcepalpha(fs))
    dst_sp_coded = pysptk.sp2mc(dst_sp, order=64, alpha=pysptk.util.mcepalpha(fs))
    print(src_sp_coded.shape, dst_sp_coded.shape, src_stft.shape)
    res = dtw(src_sp_coded, dst_sp_coded)
    path = res.get_warping_path(target='query')
    src_stft_aligned = src_stft[path]
    src_stft = invers_stft(src_stft_aligned)
    visualize(src)
    visualize(src_stft)




if __name__=='__main__':
    test2(src_path, dst_path)
    # test3(src_path, dst_path)