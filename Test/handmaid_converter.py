import librosa
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import tensorflow
import pyworld
import simpleaudio
from librosa import display
fs = sr = 24000
import noisereduce


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def wav_file_loader(wav_path, sr):
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

def no_sound_trimming(wav, top_db=35):
    intervals = librosa.effects.split(wav, top_db=top_db, ref=np.mean, frame_length=256, hop_length=256)
    non_intervals = no_sound_interval(intervals, wav.shape[0])
    res_wav = librosa.effects.remix(wav, intervals=intervals)
    return res_wav, non_intervals

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
    
def normalize2(wav):
    print(np.max(wav), np.min(wav))
    return (wav-np.min(wav))/(np.max(wav) - np.min(wav))

def visualize(data, title1=None, title2=None, title3=None):
    plt.plot(data)
    if title1 is not None: plt.title(title1)
    plt.show()
    plt.plot(data.T)
    if title1 is not None: plt.title(title1)
    plt.show()
    if title2 != '':
        display.specshow(np.array(data, dtype=np.float32))
        if title2 is not None: plt.title(title2)
        plt.show()
    if title3 != '':
        if title3 is not None: plt.title(title3)
        plt.imshow(data)
        plt.show()
    
def get_wav(wav_num, dst_num):
    data_dir = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/ProductionModel/DataStore2/parallel100'
    wav_num = str(wav_num).zfill(3)
    dst_num = str(dst_num).zfill(3)
    src_file_name = '/127_parallel100/127_{}.wav'.format(wav_num)
    dst_file_name = '/{}_parallel100/{}_{}.wav'.format(dst_num, dst_num, wav_num)
    src_wav = wav_file_loader(data_dir + src_file_name, sr)
    dst_wav = wav_file_loader(data_dir + dst_file_name, sr)
    return src_wav, dst_wav

def get_pyworld(wav):
    f0, timeaxis = pyworld.harvest(wav, fs)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    return f0, sp, ap

def get_mel(wav):
    mel = librosa.feature.melspectrogram(wav, sr=sr, n_fft=pyworld.get_cheaptrick_fft_size(fs, 71.0),
                                         hop_length=librosa.time_to_samples(0.005, fs))
    return mel

def test1():
    wav_num = 1
    dst_num = 93
    src_wav, dst_wav = get_wav(wav_num, dst_num)
    src_wav, _ = no_sound_trimming(src_wav)
    dst_wav, _ = no_sound_trimming(dst_wav)
    src_f0, src_sp, src_ap = get_pyworld(src_wav)
    dst_f0, dst_sp, dst_ap = get_pyworld(dst_wav)
    src_sp = normalize2(src_sp)
    dst_sp = normalize2(dst_sp)
    visualize(src_sp.T[:150])
    visualize(dst_sp.T[:150])
    visualize(src_sp.T[:150].T)
    visualize(dst_sp.T[:150].T)
    src_mel = get_mel(src_wav)
    dst_mel = get_mel(dst_wav)
    src_mel = normalize2(src_mel)
    dst_mel = normalize2(dst_mel)
    visualize(src_mel)
    visualize(dst_mel)
    visualize(src_mel.T)
    visualize(dst_mel.T)

def test2():
    for wav_num in range(1, 101):
        for dst_num in [93, 84]:
            src_wav, dst_wav = get_wav(wav_num, dst_num)
            src_wav, _ = no_sound_trimming(src_wav)
            dst_wav, _ = no_sound_trimming(dst_wav)
            src_f0, src_sp, src_ap = get_pyworld(src_wav)
            dst_f0, dst_sp, dst_ap = get_pyworld(dst_wav)
            # src_sp = normalize2(src_sp)
            # dst_sp = normalize2(dst_sp)
            visualize(src_sp.T, title1='src wav_num :' + str(wav_num), title2='', title3='')
            visualize(dst_sp.T, title1='dst {} wav_num :'.format(dst_num) + str(wav_num), title2='', title3='')


def good_wav(sp):
    for i in range(sp.shape[0]):
        time_sp = sp[i]
        for j in range(time_sp.shape[0]):
            sp_t = time_sp[j]

def normalizer(data):
    data = np.log(data) / np.log(10 ** 10)
    return data

def test3():
    for wav_num in range(1, 101):
        for dst_num in [93, 84]:
            src_wav, dst_wav = get_wav(wav_num, dst_num)
            dst_wav, _ = no_sound_trimming(dst_wav)
            wave_play(dst_wav)
            dst_f0, dst_sp, dst_ap = get_pyworld(dst_wav)
            # dst_sp = normalize2(dst_sp)
            # dst_sp[:, 256:] = 1e-10
            # dst_sp += 1e-10
            # dst_sp *= 10000
            dst_ap = normalize2(dst_ap)
            # dst_ap += 1e-10
            # dst_ap *= 10000
            wav = pyworld.synthesize(dst_f0, dst_sp, dst_ap, fs)
            wave_play(wav)

def normalize9(data, power=10):
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * (10 ** power - 1)

def log_normalize(data, power=10):
    data = normalize9(data, power)
    data = data + 1 # data range is 1 to 10 ** 10
    data = np.log(data) / np.log(10 ** power)
    return data

def inv_log_normalize(data, power=10):
    data = (10 ** power) ** data
    data = data - 1
    data = data / (10 ** power - 1)
    return data

#問題はlogmelで1以下の存在 >> 0以下になる。 1でも大丈夫っぽい。
#できれば1 < のオーダーに直して、正規化したい。
#10以上のデータを正確に再現したい >> 10以上のデータが大事。

def test4():
    for wav_num in range(1, 101):
        for dst_num in [93, 84]:
            src_wav, dst_wav = get_wav(wav_num, dst_num)
            dst_wav, _ = no_sound_trimming(dst_wav)
            wave_play(dst_wav)
            dst_f0, dst_sp, dst_ap = get_pyworld(dst_wav)

            # ex = 0
            # dst_sp = dst_sp * 10**ex
            # print(np.max(dst_sp))
            # dst_sp = log_normalize(dst_sp, power=10 + ex)
            # visualize(dst_sp)
            # dst_sp = inv_log_normalize(dst_sp, power= 10 + ex)
            dst_sp[np.where(dst_sp < 10)] = 1
            wav = pyworld.synthesize(dst_f0, dst_sp, dst_ap, fs)
            wave_play(wav)
            
def test5():
    for wav_num in range(1, 101):
        for dst_num in [93, 84]:
            src_wav, dst_wav = get_wav(wav_num, dst_num)
            dst_wav, _ = no_sound_trimming(dst_wav)
            wave_play(dst_wav)
            dst_mel = get_mel(dst_wav).T
            print(np.max(dst_mel), np.min(dst_mel))
            # dst_mel = log_normalize(dst_mel)
            visualize(np.log(dst_mel))
            # dst_mel = inv_log_normalize(dst_mel)
            # wav = librosa.feature.inverse.mel_to_audio(dst_mel.T, n_fft=pyworld.get_cheaptrick_fft_size(fs, 71.0),
            #                               hop_length=librosa.time_to_samples(0.005, sr), sr=sr)
            # wave_play(wav)

#mel >> 10 >> 10 ** 10 ** 12 order
#sp >> 1e-10 >> 10 ** 10 order

#clipはしちゃだめ

def _batch_parser_normalize95(input):
    input = normalize2(input)
    input = input * (10.0 ** 10 - 1) + 1
    input = input / (10.0 ** 5)
    input = np.log(input) / np.log(10.0) / 5.0
    return input

def _inv_normalize95(input):
    input = input * np.log(10.0) * 5.0
    input = np.e ** input
    input = (10 ** 5) * input
    input = (input - 1 )/ (10.0 ** 10 -1)
    return input

def test6():
    for wav_num in range(1, 101):
        for dst_num in [93, 84]:
            src_wav, dst_wav = get_wav(wav_num, dst_num)
            dst_wav, _ = no_sound_trimming(dst_wav)
            dst_f0, dst_sp, dst_ap = get_pyworld(dst_wav)
            dst_sp = _batch_parser_normalize95(dst_sp)
            visualize(dst_sp)
            dst_sp = _inv_normalize95(dst_sp)
            dst_ap = normalize2(dst_ap)
            wav = pyworld.synthesize(dst_f0, dst_sp, dst_ap, fs)
            wave_play(wav)

def size_check():
    value = 0.1
    pA = 1 + value
    pA = np.log(pA) / np.log(10 ** 10) #1e-10 < 1e10を 0<1に正規化
    pB = np.log(value) / np.log(10 ** 10) #1e-10 < 1e10を　-1 < 1に正規化
    print(pA)
    print(pB)

if __name__=='__main__':
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # size_check()
    test6()