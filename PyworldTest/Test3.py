#harvestのfloorとceilを変えた時の違い
from scipy.io import wavfile
import librosa
import pyworld
from utils import *
src_file = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/CycleGanConverterExample/TargetVoices/jvs093/parallel100/wav24kHz16bit/VOICEACTRESS100_025.wav'
sr = fs = 24000

def harvest_test():
    rate, wav = wavfile.read(src_file)
    wav = wav.astype(np.float64)
    visualize(wav)
    f0_0, t_0 = pyworld.harvest(wav, fs)
    #無音領域には関係ないが、alignmentには関係ある
    f0_1, t_1 = pyworld.harvest(wav, fs,  f0_floor = 20.0, f0_ceil = 3500.0)
    sp0 = pyworld.cheaptrick(wav, f0_0, t_0, sr)
    ap0 = pyworld.d4c(wav, f0_0, t_0, sr)
    sp1 = pyworld.cheaptrick(wav, f0_1, t_1, sr)
    ap1 = pyworld.d4c(wav, f0_1, t_1, sr)
    visualize(f0_0)
    visualize(f0_1)
    syn_wav1 = pyworld.synthesize(f0_0, sp0, ap0, fs)
    syn_wav2 = pyworld.synthesize(f0_1, sp1, ap1, fs)
    visualize(syn_wav1)
    visualize(syn_wav2)
    f0_0, t_0 = pyworld.harvest(wav, fs)
    sp0 = pyworld.cheaptrick(wav, f0_0, t_0, sr)
    ap0 = pyworld.d4c(wav, f0_0, t_0, sr)


#Trimming and converting Test
def zero_cut_reconstruct_test():
    rate, wav = wavfile.read(src_file)
    wav = wav.astype(np.float64)
    wave_play(wav)
    visualize(wav)
    intervals = librosa.effects.split(wav, top_db=40, ref=np.mean, frame_length=256, hop_length=256)
    non_intervals = no_sound_interval(intervals, len(wav))
    for start, end in non_intervals:
        print(start, end)
        wav[start: end] = 0.0
    wave_play(wav, fs)
    visualize(wav)

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

if __name__=='__main__':
    # harvest_test()
    zero_cut_reconstruct_test()