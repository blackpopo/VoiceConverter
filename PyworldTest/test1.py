from utils import *
import pyworld
from pprint import pprint
import numpy as np
import librosa
import tensorflow as tf
import DP
from nnmnkwii.preprocessing import trim_zeros_frames
from nnmnkwii.preprocessing.alignment import DTWAligner

wave_path = 'TargetVoices/jvs093/parallel100/wav24kHz16bit/VOICEACTRESS100_025.wav'
wave_dir = 'TargetVoices/jvs093/parallel100/wav24kHz16bit/'
fs = sr = 24000
src_path = 'SourceVoices/jvs034/parallel100/wav24kHz16bit/VOICEACTRESS100_025.wav'
# src_path = '../test-voices/test1.wav'
dst_path = 'TargetVoices/jvs093/parallel100/wav24kHz16bit/VOICEACTRESS100_025.wav'


def dio_stonemask_harvest(wave_path):
    wav = load_wave(wave_path)
    wav = wav.astype(np.float64)
    print(wav.shape)
    visualize(wav)
    dio_f0, tdio = pyworld.dio(wav, fs=fs, frame_period=3.0)
    stone_f0 = pyworld.stonemask(wav, dio_f0, tdio, fs=fs)
    visualize(dio_f0)
    visualize(stone_f0)
    print("tdio", tdio.shape)
    harvest_f0, thrv = pyworld.harvest(wav, fs=fs, frame_period=3.0)
    stone_f0_1 = pyworld.stonemask(wav, harvest_f0, thrv, fs=fs)
    print("Harvest {}, Stone{}".format(harvest_f0, stone_f0_1))
    visualize(harvest_f0)
    visualize(stone_f0_1)

def first_synthesize(wave_path):
    wav = load_wave(wave_path)
    wav = wav.astype(np.float64)
    f0, t = pyworld.harvest(wav, fs=fs)
    sp = pyworld.cheaptrick(wav, f0, t, fs)
    ap = pyworld.d4c(wav, f0, t, fs)
    syn_wav = pyworld.synthesize(f0, sp, ap, fs)
    return syn_wav

def f0_change(src_wav_path, dst_wave_path):
    src = load_wave(src_wav_path)
    dst = load_wave(dst_wave_path)
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    src_f0, src_t = pyworld.harvest(src, fs=fs)
    dst_f0, dst_t = pyworld.harvest(dst, fs=fs)
    visualize(src_f0[:])
    visualize(dst_f0[:])
    dist_0, _ = fastdtw.fastdtw(src_f0, dst_f0)
    # print(src_f0.shape, dst_f0.shape, src_t.shape, dst_t.shape)
    #916, 1005, 916, 1005
    length = min(src_f0.shape[0], dst_f0.shape[0])
    src_sp = pyworld.cheaptrick(src, src_f0, src_t, fs=fs)
    dst_sp = pyworld.cheaptrick(dst, dst_f0, dst_t, fs=fs)
    dist_1, _ = fastdtw.fastdtw(src_sp, dst_sp)
    # visualize(src_sp)
    # visualize(dst_sp)
    # src_sp_coded = pyworld.code_spectral_envelope(src_sp, fs, 128)
    # dst_sp_coded = pyworld.code_spectral_envelope(dst_sp, fs, 64)
    # visualize(src_sp_coded)
    # visualize(dst_sp_coded)
    # print(src_sp.shape, dst_sp.shape)
    # print(dst_sp_coded.shape, src_sp_coded.shape)
    #(916, 513), (1005, 513)
    visualize(src_sp[:, 0])
    visualize(dst_sp[:, 0])
    src_ap = pyworld.d4c(src, src_f0, src_t, fs=fs)
    dst_ap = pyworld.d4c(dst, dst_f0, dst_t, fs=fs)
    #pyworld synthesize(f0, sp, ap, fs)
    #(916, 513), (1005, 513)
    # print(src_ap.shape, dst_ap.shape)
    visualize(src_ap[:, 0])
    visualize(dst_ap[:, 0])
    dist_2, _ = fastdtw.fastdtw(src_ap, dst_ap)
    # syn_wav = pyworld.synthesize(src_f0[:length], dst_sp[:length, :], dst_ap[:length, :], fs)
    # syn_wav = pyworld.synthesize(dst_f0[:length], src_sp[:length, :], dst_ap[:length, :], fs)
    # syn_wav = pyworld.synthesize(dst_f0[:length], dst_sp[:length, :], src_ap[:length, :], fs)
    # return syn_wav
    print(dist_0, dist_1, dist_2)

def test2(wavw_path):
    wav = load_wave(wave_path)
    wav = wav.astype(np.float64)
    f0, t = pyworld.harvest(wav, fs)
    sp = pyworld.cheaptrick(wav, f0, t, fs)
    ap = pyworld.d4c(wav, f0, t, fs)
    sp_coded = pyworld.code_spectral_envelope(sp, fs, 128)
    sp_decoded = pyworld.decode_spectral_envelope(sp_coded, fs, pyworld.get_cheaptrick_fft_size(fs))
    ap_coded = pyworld.code_aperiodicity(ap, fs)
    ap_decoded =  pyworld.decode_aperiodicity(ap_coded, fs, pyworld.get_cheaptrick_fft_size(fs))
    syn_wav = pyworld.synthesize(f0, sp_decoded, ap_decoded, fs)
    return syn_wav

def test3(wave_path):
    wav = load_wave(wave_path)
    wav = wav.astype(np.float64)
    stft = librosa.stft(wav)
    spectrograms = tf.abs(stft)
    S = librosa.feature.melspectrogram(wav, sr)
    # S, phase = librosa.magphase(stft)
    # Sdb = librosa.amplitude_to_db(S)
    f0, t = pyworld.harvest(wav, fs)
    sp = pyworld.cheaptrick(wav, f0, t, fs)
    mfcc = librosa.feature.mfcc(wav)
    visualize(f0)
    visualize(stft)
    visualize(sp)
    visualize(S)
    visualize(mfcc)

def sub(x, y):
    return x - y

import fastdtw
def test4(src_path, dst_path):
    src = load_wave(src_path)
    dst = load_wave(dst_path)
    print(src.shape, dst.shape)
    # src = DP.align(src, dst, abs(src.shape[0]-dst.shape[0]), 'euclidean', src)
    # src = src.astype(np.float64)
    # dst = dst.astype(np.float64)
    # src_f0, src_t = pyworld.harvest(src, fs=fs)
    # dst_f0, dst_t = pyworld.harvest(dst, fs=fs)
    visualize(src)
    visualize(dst)

def collect_features(wav):
    # f0, timeaxis = pyworld.dio(wav, fs, frame_period=5.0)
    # f0 = pyworld.stonemask(wav, f0, timeaxis, fs)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period=5.0)
    spectrogram = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    spectrogram = trim_zeros_frames(spectrogram)
    return spectrogram

def alignment(x, y):
    x_aligned, y_aligned = DTWAligner().transform((x[np.newaxis, :, :], y[np.newaxis, :, :]))
    x_aligned, y_aligned = x_aligned[:, :, 1:], y_aligned[:, :, 1:]
    x_aligned = np.ascontiguousarray(x_aligned, dtype=np.float64)
    y_aligned = np.ascontiguousarray(y_aligned, dtype=np.float64)
    x_aligned = x_aligned.squeeze(0)
    y_aligned = y_aligned.squeeze(0)
    return x_aligned, y_aligned

def test5(src_wav_path, dst_wave_path):
    src = load_wave(src_wav_path)
    dst = load_wave(dst_wave_path)
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    src_f0, src_t = pyworld.harvest(src, fs=fs)
    dst_f0, dst_t = pyworld.harvest(dst, fs=fs)
    src_sp = pyworld.cheaptrick(src, src_f0, src_t, fs=fs)
    dst_sp = pyworld.cheaptrick(dst, dst_f0, dst_t, fs=fs)
    src_ap = pyworld.d4c(src, src_f0, src_t, fs=fs)
    dst_ap = pyworld.d4c(dst, dst_f0, dst_t, fs=fs)
    src_sp = trim_zeros_frames(src_sp)
    dst_sp = trim_zeros_frames(dst_sp)
    src_ap, dst_ap = alignment(src_ap, dst_ap)
    src_sp, dst_sp = alignment(src_sp, dst_sp)
    src_wav = pyworld.synthesize(src_f0, src_sp, src_ap, fs)
    dst_wav = pyworld.synthesize(dst_f0, dst_sp, dst_ap, fs)
    visualize(src_wav)
    visualize(dst_wav)
    wave_play(src_wav)
    wave_play(dst_wav)


if __name__=='__main__':
    # wave_play(src_path)
    # wave_play(dst_path)
    # wave_play(f0_change(src_path, dst_path))
    # dio_stonemask_harvest(wave_path)
    # wave_play(first_synthesize(wave_path))
    # wav_padding(load_wave(dst_path, sr), sr, 5.0)
    # wave_play(test2(dst_path))
    # test3(dst_path)
    # test4(src_path, dst_path)
    # f0_change(src_path, dst_path)
    test5(src_path, dst_path)