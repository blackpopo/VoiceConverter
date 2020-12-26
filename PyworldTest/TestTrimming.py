import numpy as np
import pyworld
from utils import *
import librosa
from scipy.fftpack import dct
from nnmnkwii.preprocessing import trim_zeros_frames

fs = sr = 24000
src_path = 'SourceVoices/jvs034/parallel100/wav24kHz16bit/VOICEACTRESS100_025.wav'
# src_path = '../test-voices/test1.wav'
dst_path = 'TargetVoices/jvs093/parallel100/wav24kHz16bit/VOICEACTRESS100_025.wav'

def ddtw(x):
    S = np.copy(x)
    for i in range(S.shape[0]-1):
        S[i] = 1.0 * x[i] - 0.25* x[i-1] + 0.25 * x[i+1]
    return S

def nnmkwii_triming(wav):
    #Works For Spectrum Data?
    wav = np.array(wav).astype(np.float32)
    wav = trim_zeros_frames(wav)
    return wav

def librosa_trimming(wav, top_db):
    sound = np.array(wav).astype(np.float32)
    wav , index = librosa.effects.trim(sound, top_db=top_db)
    return wav

def original_trimming(wav):
    wav = np.array(wav).astype(np.float32)
    sound = wav[:].reshape((-1, 2 ** 6))
    print(sound.shape)
    convert = dct(sound, norm='ortho')
    print(convert.shape)
    abs_converted = np.abs(convert) #[ここでデータの抜き出し
    where = np.where(np.max(abs_converted, axis=1) < 512)[0]
    wav = np.delete(convert, where, axis=0)
    return wav



if __name__=='__main__':
    wav = load_wave(dst_path, sr)
    wave_play(wav, fs)
    visualize(wav)
    res_wav = librosa_trimming(wav)
    visualize(res_wav)
    wave_play(res_wav)

