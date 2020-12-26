import numpy as np
import pyworld
import matplotlib.pyplot as plt
from scipy.io import wavfile
fs = 24000
import simpleaudio
# ap はcode すれば3次元でいいらしい？
import  os
from tqdm import trange

def world_decompose(wav, fs, frame_period = 5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    #なんでf0_floor　とf0_ceilの値を変えてんの？
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    #声の聞こえる範囲だけど、雑音はいるから却下
#     f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 20.0, f0_ceil = 3500.0)
    sp_513 = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    visualize(sp_513)
    visualize(sp_513[:, :256])
    visualize(sp_513[:, 256])
    visualize(sp_513[:, :128])
    visualize(sp_513[:, 128])
    ap_513 = pyworld.d4c(wav, f0, timeaxis, fs)
    visualize(ap_513)
    visualize(ap_513[:, 0])
    visualize(ap_513[:, 1])
    visualize(ap_513[:, 2])
    visualize(ap_513[:, 3])
    visualize(ap_513[:, 4])
    sp_256 = pyworld.code_spectral_envelope(sp_513, fs, 256)
    visualize(sp_256)
    ap_3 = pyworld.code_aperiodicity(ap_513, fs)
    #24000Hzなら、3次元あれば十分なの？
    num = pyworld.get_num_aperiodicities(fs)
    visualize(ap_3)
    for i in range(num):
        visualize(ap_3[:, i])
    sp_128 = pyworld.code_spectral_envelope(sp_513, fs, 128)
    visualize(sp_128)

def makenoise(data, shape, scale):
    noise = np.random.normal(loc=1.0, scale=scale, size=shape)
    return data * noise
#ノイズをかけてもできるだけきれいに戻る条件を見つける
#2**18の場合誤差85%,2**16で90%,2**14で95%と仮定する

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
    return np.max(np.abs(data)), data / np.max(np.abs(data))

def log_normalize(x):
    x = x/np.max(np.abs(x))
    x = -np.log10(x)
    return x/np.max(x)

def rev_log_normalize(x, base=10):
  x = -(x * base)  # 10で声の大きさが変わる 093の場合は13が最適
  return 10 ** x

def rev_normalize(x, x_std, x_mean, axis = None):
    return x * x_std + x_mean

#definition L1 loss function
def L1_loss(y_true,y_pre):
    return np.mean(np.abs(y_true-y_pre))

def world_test(wav, frame_period):
    wav = wav.astype(np.float64)
    wave_play(wav, fs=24000)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)
    sp_513 = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap_513 = pyworld.d4c(wav, f0, timeaxis, fs)
    scale, sp_513 = normalize(sp_513)
    scale, ap_513 = normalize(ap_513)
    # visualize(sp_513[:, 256])

    # sp_513は6割以上あってればOK
    #64だと声が変わる
    #128だとなんか変
    # sp_256は8割以上あってればOK
    # sp_513_noised = makenoise(sp_513, sp_513.shape, 0.3)
    # print(L1_loss(sp_513_noised, sp_513))
    # sp_513 = sp_513_noised

    #ap_513はほぼかんけーねー
    # ap_513 = makenoise(ap_513, ap_513.shape, 0.1)
    #f0は正解率97.5%が必要！
    # f0 /= 800
    # f0 = f0[:256]
    # f0_noised = makenoise(f0, f0.shape, 0.0)
    # print(L1_loss(f0[:256], f0_noised[:256]))
    # f0 = 800 * f0_noised
    sp_256 = pyworld.code_spectral_envelope(sp_513, fs, 256)
    # # 97.5 % の誤差率に抑えられればOK
    scale, sp_256 = normalize(sp_256)

    sp_256_noised = makenoise(sp_256, sp_256.shape, scale=0.01)
    print(L1_loss(sp_256[:512], sp_256_noised[:512]))
    sp_256 = scale * sp_256_noised
    sp_513 = pyworld.decode_spectral_envelope(sp_256, fs, pyworld.get_cheaptrick_fft_size(fs))
    wav = pyworld.synthesize(f0, sp_513, ap_513, fs, frame_period=frame_period)
    wave_play(wav, fs=24000)

#まとめると
#f0は97.5%の正解率が必要
#sp_256なら8割以上の正解率が必要
#sp_513なら6割以上の正解率が必要
#frame priod　は5~10の範囲が限界。>> 10のほうが、まだかすれない。あんま関係ない。多く時系列データを取れたほうがいいから10にしとくか？

def visualize(data):
    plt.plot(data)
    plt.show()

def std_test(wav):
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, f0_floor=71.0, f0_ceil=800.0)
    sp_513 = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap_513 = pyworld.d4c(wav, f0, timeaxis, fs)
    sp_513 = log_normalize(sp_513)
    visualize(sp_513)
    print(sp_513)
    sp_513 = rev_log_normalize(sp_513, base=13)
    # ap_513 = log_normalize(ap_513)
    # ap_513 = rev_log_normalize(ap_513)
    wav = pyworld.synthesize(f0, sp_513, ap_513, fs)
    wave_play(wav, fs)
    # with open('sp_mean_mean.txt', 'r') as f: #meanの平均の読み取り
    #     x_mean = f.readlines()
    # x_mean = np.array([float(x.strip('\n')) for x in x_mean])
    # with open('sp_std_mean.txt', 'r') as f: #stdの平均の読み取り
    #     x_std = f.readlines()
    # x_std = np.array([float(x.strip('\n')) for x in x_std])

    # sp_513_noised = makenoise(sp_513, sp_513.shape, scale=0.005) #noiseをかけて正確に聞けるか？
    # print(L1_loss(sp_513[:256], sp_513_noised[:256])) #noise をかけた時のL1 loss
    # ap_513 = std_normalize(ap_513, axis=0)

    # sp_513 = rev_normalize(sp_513, x_std, x_mean) #spの復元
    # wav = pyworld.synthesize(f0, sp_513, ap_513, fs)
    # wave_play(wav, fs=24000)



def read093data():
    wav_source = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/CycleGanConverterExample/TargetVoices/jvs093/parallel100/wav24kHz16bit'
    files = os.listdir(wav_source)
    sp_means = list()
    sp_stds = list()
    for i in trange(len(files)):
        file = files[i]
        _, wav = wavfile.read(os.path.join(wav_source,file))
        wav = wav.astype(np.float64)
        f0, timeaxis = pyworld.harvest(wav, fs, f0_floor=71.0, f0_ceil=800.0)
        sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
        ap = pyworld.d4c(wav, f0, timeaxis, fs)
        sp, sp_std, sp_mean = normalize(sp)
        sp_means.append(sp_mean)
        sp_stds.append(sp_std)
    sp_means = np.array(sp_means)
    sp_stds = np.array(sp_stds)
    _ , sp_mean_std , sp_mean_mean = normalize(sp_means)
    _, sp_std_std ,  sp_std_mean = normalize(sp_stds)

    def np2str(np_list):
        return [str(num) for num in np_list]

    sp_mean_std = np2str(sp_mean_std)
    sp_mean_mean = np2str(sp_mean_mean)
    sp_std_mean = np2str(sp_std_mean)
    sp_std_std = np2str(sp_std_std)

    with open('sp_mean_std.txt', 'w') as f:
        f.writelines('\n'.join(sp_mean_std))
    with open('sp_mean_mean.txt', 'w') as f:
        f.writelines('\n'.join(sp_mean_mean))
    with open('sp_std_std.txt', 'w') as f:
        f.writelines('\n'.join(sp_std_std))
    with open('sp_std_mean.txt', 'w') as f:
        f.writelines('\n'.join(sp_std_mean))

def log_noise_test(wav):
    wave_play(wav)
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, f0_floor=71.0, f0_ceil=800.0)
    sp_513 = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap_513 = pyworld.d4c(wav, f0, timeaxis, fs)
    sp_513 = log_normalize(sp_513)
    sp_513[:, 256:] = 1e-6
    sp_513_noised = makenoise(sp_513, sp_513.shape, scale=0.05)
    print(L1_loss(sp_513[256:513], sp_513_noised[256:513]))
    sp_513 = rev_log_normalize(sp_513_noised, base=13)
    sp_513[:, 256:] = 1e-6
    wave_play(pyworld.synthesize(f0, sp_513, ap_513, fs), fs)

if __name__=='__main__':
    wav_source = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/093_aligned_experiment/093_093_001.wav'
    _, wav = wavfile.read(wav_source)
    wav = wav.astype(np.float64)
    # world_test(wav, frame_period = 5)
    # std_test(wav)
    # read093data()
    log_noise_test(wav)