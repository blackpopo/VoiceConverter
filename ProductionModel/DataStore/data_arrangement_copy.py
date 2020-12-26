import os
from tqdm import trange, tqdm
import librosa
from nnmnkwii.preprocessing import trim_zeros_frames
import numpy as np
import pyworld
import pysptk
from dtwalign import dtw
from collections import defaultdict

def librosa_remixing(wav, top_db):
    wav = wav.astype(np.float64)
    res_wav = librosa.effects.remix(wav, intervals=librosa.effects.split(wav, top_db=top_db, ref=np.mean, frame_length=256,  hop_length=256))
    return res_wav

sr = fs = 24000

def alignment(src_wavs, dst_wav, wav_num, save_path):
    dst = librosa_remixing(dst_wav, top_db=50)
    dst_f0, dst_t = pyworld.harvest(dst, fs=fs)
    dst_sp = pyworld.cheaptrick(dst, dst_f0, dst_t, fs=fs)
    #この時点でtf Recordに書き込んでしまおう！どうせ分割してもtfRecordでくっつくし
    dst_sp_segmap = np_create_segmap(dst_sp)
    dst_sp = normalize(dst_sp)
    dst_sp_coded = pysptk.sp2mc(dst_sp, order=32, alpha=pysptk.util.mcepalpha(fs))
    dst_sp_coded = trim_zeros_frames(dst_sp_coded)
    for who, src_wav in tqdm(src_wavs):
        res = src_alignment(src_wav, dst_sp_coded)
        if res is not None:
            src_f0, src_sp = res
            src_sp = normalize(src_sp)


def src_alignment(src_wav, dst_sp_coded):
    src = librosa_remixing(src_wav, top_db=50)
    src_f0, src_t = pyworld.harvest(src, fs=fs)
    src_sp = pyworld.cheaptrick(src, src_f0, src_t, fs=fs)
    # src_ap = pyworld.d4c(src, src_f0, src_t, fs=fs)
    src_sp_coded = pysptk.sp2mc(src_sp, order=32, alpha=pysptk.util.mcepalpha(fs))
    src_sp_coded = trim_zeros_frames(src_sp_coded)
    try:
        res = dtw(src_sp_coded, dst_sp_coded, step_pattern="typeIb")
        path = res.get_warping_path(target='query')
        src_f0_aligned = src_f0[path]
        # src_ap_aligned = src_ap[path, :]
        src_sp_aligned = src_sp[path, :]
        return src_f0_aligned, src_sp_aligned
    except:
        return None

def np_create_segmap(image):
    mean = np.mean(image)
    segmap = np.zeros_like(image)
    segmap[np.where(image > mean)] = 1
    return segmap

def normalize(array):
    return array / np.max(np.abs(array))

def collect_wavs(target_number, wav_num, datalist, save_path):
    source_wavs = list()
    target_wav = None
    for who, path in tqdm(datalist):
        if who == target_number:
            target_wav = librosa.core.load(path, sr=sr)[0]
            target_wav = target_wav.astype(np.float64)
        else:
            source_wavs.append((who, librosa.core.load(path, sr=sr)[0]))
    alignment(source_wavs, target_wav, wav_num, save_path)

def VoiceMakers(target_number):
    source_base_folder = "C:/Users/Atsuya/Documents/Sites/Voice/jvs_ver1/jvs_ver1"
    folders = os.listdir(source_base_folder)
    paths = collect_same_num_path(source_base_folder, folders)
    save_path = os.path.join('C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices', target_number + '_aligned2')
    for wav_num, data_list in tqdm(paths.items()):
        collect_wavs(target_number, wav_num, data_list, save_path)
    
    
def collect_same_num_path(source_base_folder, folders):
    collect_path = defaultdict(list)
    for folder in folders:
        who = folder[-3:]
        files = os.listdir(os.path.join(source_base_folder, folder, 'parallel100', 'wav24kHz16bit'))
        for file in files:
            num = file.split('.')[0][-3:]
            collect_path[num].append((who, os.path.join(source_base_folder, folder, 'parallel100', 'wav24kHz16bit', file)))
    return collect_path

if __name__=='__main__':
    VoiceMakers("084")