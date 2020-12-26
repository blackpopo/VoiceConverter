#HorizontalFilp
#ScaleAugmentation
#RandomCrop
#Per-pixel Mean Subtraction
#Mean Subtraction
# from scipy.fftpack import rfft, irfft
# from test_utils import *
from scipy.io import wavfile
from scipy import signal
import numpy as np
samplerate = fs = 24000

# def patch_lowpass(src_wav, tgt_wav):
#     freq = np.random.randint(20000, 35000)
#     rsrc = lowpass(src_wav, freq)
#     rtgt = lowpass(tgt_wav, freq)
#     return (rsrc, rtgt)

# def patch_highpass(src_wav, tgt_wav):
#     freq = np.random.randint(5000, 30000)
#     rsrc = highpass(src_wav, freq)
#     rtgt = highpass(tgt_wav, freq)
#     return (rsrc, rtgt)

# def patch_bandpass(src_wav, tgt_wav):
#     low_freq = np.random.randint(10000, 35000)
#     high_freq = np.random.randint(5000, 30000)
#     if low_freq < high_freq:
#         temp = low_freq
#         low_freq = high_freq
#         high_freq = temp
#         if low_freq - high_freq < 10000:
#             low_freq = low_freq + 10000
#     print(low_freq, high_freq)
#         #low_passのほうがhighpassよりおおきくなる
#     rsrc = lowpass(highpass(src_wav, high_freq), low_freq)
#     rtgt = lowpass(highpass(tgt_wav, high_freq), low_freq)
#     return rsrc, rtgt
#
# #使えん（笑）
# def patch_stopbandpass(src_wav, tgt_wav):
#     low_freq = np.random.randint(10000, 35000)
#     high_freq = np.random.randint(5000, 30000)
#     if low_freq > high_freq:
#         temp = low_freq
#         low_freq = high_freq
#         high_freq = temp
#         #low_passのほうがhighpassより小さくなる。
#         if high_freq - low_freq < 10000:
#             high_freq = high_freq + 10000
#     print(low_freq, high_freq)
#     rsrc = lowpass(highpass(src_wav, high_freq), low_freq)
#     rtgt = lowpass(highpass(tgt_wav, high_freq), low_freq)
#     visualize(rsrc)
#     return rsrc, rtgt

#
# def lowpass(wav, freq):
#     yf = rfft(wav)
#     yf[freq:] = 0.0
#     wav = irfft(yf)
#     return wav
# #
# # def highpass(wav, freq):
# #     yf = rfft(wav)
# #     yf[:freq] = 0.0
# #     wav = irfft(yf)
# #     return wav
#
# def gaussean_noise(wav):
#     noise = np.random.normal(loc=1, scale=0.2, size=len(wav))
#     return wav * noise
#
# #shift 300 ~ 100
# def leftshit(wav, shift):
#     yf = rfft(wav)
#     yf = np.roll(yf, -shift)
#     yf[-shift:] = 0.0
#     wav = irfft(yf)
#     return wav
#
# #shift 100 ~ 300
# def rightshit(wav, shift):
#     yf = rfft(wav)
#     yf = np.roll(yf, shift)
#     yf[:shift] = 0.0
#     wav = irfft(yf)
#     return wav
#
#
# #p 広域を強調する強さ 1.2 ~ 0.8
# def pre_emphasis(wave, p=0.5):
#     # 係数 (1.0, -p) のFIRフィルタを作成
#     return signal.lfilter([1.0, -p], 1, wave)
#
# #信号を平準化します。 信号の高周波数成分を維持しつつ平準化したいときに効果的なフィルタ >> windowlength 10 ~ 100
# def savgol_filter(wav, window_length):
#     return signal.savgol_filter(wav, window_length, 1)
#
# #10~50の奇数
# def median_filter(wav, kernel_size=None):
#     return signal.medfilt(wav, kernel_size)

# #全くわからん
# def sos_filter(wav):
#     sos = signal.ellip(1, 0.1, 10, 0.5, output='sos')
#     return  signal.sosfilt(sos, wav)

# #pが強さ？ >> 使えそう？ p感けーねー　#フィルターかけたやつに合わせるみたい？ 使えん
# def filtfilt(wav, p=1.5):
#     return signal.filtfilt([1.0, -p], 1, wav)

#使える 0.9 ~ 1.1 * len(wav)
# def resamplefilter(wav, num):
#     return signal.resample(wav, num)
#
# # def inverse_wave(wav):
#     return wav[::-1]

# #使えん！
# def reverse(wav, start, end):
#     visualize(wav)
#     yf = rfft(wav)
#     reverse_indices = range(start, end)[::-1]
#     yf = yf[reverse_indices]
#     wav = irfft(yf)
#     visualize(wav)
#     return wav
#
# #なんか、早口になる　＞＞　使えん
# def decimate_filter(wav, q=2):
#     return signal.decimate(wav, q)
#
# #なんかスケールが違う気がする　>> 使えん
# def symfilter(wav, c0, z1):
#     return signal.symiirorder1(wav, c0, z1)

# def zero_factor(wav, zero_list):
#     for i in zero_list:
#         wav[i] = 0.0
#     return wav

import pyworld
import os
from tqdm import trange


def world_decompose(wav, fs, frame_period = 5.0, to256=False):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    #なんでf0_floor　とf0_ceilの値を変えてんの？
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    #声の聞こえる範囲だけど、雑音はいるから却下
#     f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 20.0, f0_ceil = 3500.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    if to256:
        coded_sp = pyworld.code_spectral_envelope(sp, fs, 256)
        coded_ap = pyworld.code_aperiodicity(ap, fs)
        print('ap is coded to {}'.format(pyworld.get_num_aperiodicities(fs)))
    return f0, timeaxis, sp, ap

if __name__=='__main__':
    # wav_source = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/093_aligned_experiment/001_093_001.wav'
    # _, wav = wavfile.read(wav_source)
    # wav = wav.astype(np.float64)

    # wave_play(wav, fs)
    # random_list = np.random.randint(0, len(wav), int(len(wav)*0.03))
    # wav = zero_factor(wav, random_list)
    #
    # wave_play(wav)

    from tqdm import trange
    test_source = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/zero_cut_experiment'
    files = os.listdir(test_source)
    # wavs = list()
    # f0s = list()
    # aps = list()
    # sps = list()
    # max_t_ap = list()
    # max_v_ap = list()
    # min_ap = list()
    for i in trange(len(files)):
        file = files[i]
        _, wav = wavfile.read(os.path.join(test_source, file))
        wav = wav.astype(np.float64)[:256]
        world_decompose(wav, fs, to256=True)
    #     f0_0, t_0 = pyworld.harvest(wav, fs)
    #     sp0 = pyworld.cheaptrick(wav, f0_0, t_0, fs)
    #     ap0 = pyworld.d4c(wav, f0_0, t_0, fs)
    #     f0s.append(f0_0)
    #     # aps.append(aps)
    #     max_t_ap.append(np.max(ap0, axis=0))
    #     max_v_ap.append(np.max(ap0, axis=1))
    #     min_ap.append(np.min(ap0))
    #     sps.append(sp0)
    # visualize(np.max(f0s, axis=1))
    # visualize(np.max(sps, axis=0))
    # # visualize(np.max(aps, axis=0))
    # visualize(np.max(np.abs(f0s)))
    # visualize(np.max(np.abs(sps)))
    # # visualize(np.max(abs(aps)))
    # visualize(max_t_ap)
    # visualize(max_v_ap)
    # visualize(min_ap)
    # print(np.max(max_t_ap), np.min(max_t_ap))
    # print(np.max(max_v_ap), np.min(max_v_ap))
    #f0は800でスケーリングすればいいや
    # [260.27046212 340.700801   228.33686585 427.97691232 285.13856288
    #  357.45150589 768.09358575 337.16504142 325.62285238 435.16385844
    #  446.70429119 310.35017023 243.4211656  445.08441245 342.93936884
    #  446.55396406 319.36568572 439.14510094 350.60064117 329.73988884
    #  158.12907871 208.89386847 264.62026743 565.34406528 463.44712808
    #  496.13681383 371.85588781 330.14760558 330.70079915 405.53554349
    #  258.06495575 284.60211822 216.10626365 285.37563077 447.32194434
    #  517.90490645 343.80763902 342.76993857 405.73720608 402.28702768
    #  469.11263838 188.33000311 336.0712947  248.80041014 301.10030159
    #  289.24818138 329.50117661 221.16846405 243.45932335 271.31904157
    #  495.65047512 258.85661756 404.33947909 395.01861821 409.29086769
    #  361.17247107 367.82903511 410.08549856 351.52894564 485.88192335
    #  607.78337453 458.48548307 380.67662626 321.20434868 423.85972988
    #  423.31832042 374.69817467 286.46157682 351.2029385  294.53506907
    #  244.97270296 350.96496656 266.66027346 186.82207403 332.40859556
    #  324.223982   257.84781939 349.49145554 216.27742185 233.13573505
    #  268.6171398  358.08410935 357.18561136 498.14884645 415.00083916
    #  238.86101804 470.27987169 328.20950187 260.8292359  468.23906919
    #  358.45069274 474.9376079  516.70794503 347.546336   547.37984471
    #  359.74878981 325.19668921 287.41202787 304.93129081 236.63977016]
    # [[6.90146218e+08 6.97138679e+08 7.19463940e+08... 5.97655142e+05
    #   5.83850090e+05 6.36857074e+05]
    #  [8.37954823e+08 8.47276200e+08 8.75807360e+08... 6.37494060e+05
    #  6.77660878e+05 6.91835365e+05]
    # [3.95980705e+08
    # 4.00646819e+08
    # 4.15654177e+08...
    # 7.68287288e+04
    # 7.93394787e+04
    # 8.02163979e+04]
    # ...
    # [3.47819348e+08
    # 3.51252406e+08
    # 3.61262198e+08...
    # 2.47991105e+05
    # 2.58225770e+05
    # 2.61685041e+05]
    # [7.30282296e+08 7.37916734e+08 7.60953164e+08... 4.02738271e+05
    #  4.42763875e+05 4.57660451e+05]
    # [1.40061401e+08
    # 1.40981239e+08
    # 1.43299566e+08...
    # 4.55661716e+04
    # 4.37120432e+04
    # 4.32440092e+04]]

    # wavs = np.array(wavs)
    # print(wavs.shape)
    # # print(np.max(wavs, axis=0))
    # # print(np.max(wavs, axis=1))
    # # print(np.min(wavs, axis=0))
    # # print(np.min(wavs, axis=1))
    # # print(np.max(np.max(wavs, axis=0)))
    # # print(np.min(np.min(wavs, axis=1)))
    # print(np.min(wavs))
    # print(np.max(wavs))

    # f0_0, t_0 = pyworld.harvest(wav, fs)
    # sp0 = pyworld.cheaptrick(wav, f0_0, t_0, fs)
    # ap0 = pyworld.d4c(wav, f0_0, t_0, fs)
    # print(np.max(np.abs(f0_0)))
    # print(np.max(np.abs(sp0)))
    # print(np.max(np.abs(ap0)))
    # # 260.2704621176534
    # # apは正規化されてるみたい
    # #spはどうしよう？
    # wav = wav / np.max(np.abs(wav))
    # # wave_play(wav, fs)
    # print(np.max(np.abs(f0_0)))
    # print(np.max(np.abs(sp0)))
    # print(np.max(np.abs(ap0)))
    # 260.2704621176534
    # 6011123447.9553385
    # 0.9999999999998851
    # print(np.max(sp0), np.max(ap0))
    # print(np.min(sp0), np.min(ap0))
    # sp0 /= np.max(np.abs(sp0))
    # ap0 /= np.max(np.abs(ap0))
    # wav = pyworld.synthesize(f0_0, sp0, ap0, fs)
    # print(np.max(sp0), np.max(ap0))
    # print(np.min(sp0), np.min(ap0))




