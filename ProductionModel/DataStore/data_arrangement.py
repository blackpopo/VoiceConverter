import os
from tqdm import trange, tqdm
import librosa
from nnmnkwii.preprocessing import trim_zeros_frames
import numpy as np
import pyworld
import pysptk
from dtwalign import dtw
from collections import defaultdict
import tensorflow as tf
import os

def _bytes_feature(value):
  """string / byte 型から byte_list を返す"""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """float / double 型から float_list を返す"""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def _int64_feature(value):
  """bool / enum / int / uint 型から Int64_list を返す"""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def parse_sp_spade_features(wav_num, src_who, dst_who, src_sp, tgt_full_sp, segmap):
    data = tf.train.Example(features=tf.train.Features(feature={
        'wav_num': _bytes_feature(wav_num.encode('utf-8')),
        'src_who': _bytes_feature(src_who.encode('utf-8')),
        'dst_who': _bytes_feature(dst_who.encode('utf-8')),
        'src_sp' : _float_feature(src_sp[:, 256]),
        'tgt_sp_full': _float_feature(tgt_full_sp),
        'segmap_full': _float_feature(segmap)
    }))
    return data.SerializeToString()

def librosa_remixing(wav, top_db):
    wav = wav.astype(np.float64)
    res_wav = librosa.effects.remix(wav, intervals=librosa.effects.split(wav, top_db=top_db, ref=np.mean, frame_length=256,  hop_length=256))
    return res_wav

sr = fs = 24000

def alignment(src_wavs, dst_wav ,wav_num, target_number, save_path):
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
            tfpath = os.path.join(save_path, 'sp_spade_dst_' + target_number + '_src_' + who + '_wav_' + wav_num)
            with tf.io.TFRecordWriter(tfpath) as writer:
                rec = parse_sp_spade_features(wav_num, who, target_number, src_sp, dst_sp, dst_sp_segmap)
                writer.write(rec)

#無音消去 >> normalize >> alignment？　無音消去 >> alignment >> normalize ?
def src_alignment(src_wav, dst_sp_coded):
    src = librosa_remixing(src_wav, top_db=50)
    src_f0, src_t = pyworld.harvest(src, fs=fs)
    src_sp = pyworld.cheaptrick(src, src_f0, src_t, fs=fs)
    src_f0 = (src_f0 - 400.0) / 800.0
    src_sp = normalize(src_sp)
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
    alignment(source_wavs, target_wav, wav_num, target_number, save_path)
    # np.savez(os.path.join(save_path, str(wav_num)), target = np.array([target_number, dst_f0, dst_sp]), source=src_aligned)
    # return dst_f0, dst_sp, src_aligned

def VoiceMakers(target_number):
    source_base_folder = "C:/Users/Atsuya/Documents/Sites/Voice/jvs_ver1/jvs_ver1"
    folders = os.listdir(source_base_folder)
    paths = collect_same_num_path(source_base_folder, folders)
    save_path ='C:/Users/Atsuya/PycharmProjects/VoiceConverter/ProductionModel/DataStore/093_084_sp_spade'
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