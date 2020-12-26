#No Sound Cuttting & Alignment &
import os
from tqdm import trange, tqdm
import librosa
from nnmnkwii.preprocessing import trim_zeros_frames
import numpy as np
import pyworld
import pysptk
from dtwalign import dtw
import tensorflow as tf
import os
from utilities import *
sr = fs = 24000

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

def librosa_remixing(wav, top_db):
    wav = wav.astype(np.float64)
    res_wav = librosa.effects.remix(wav, intervals=librosa.effects.split(wav, top_db=top_db, ref=np.mean, hop_length=1024))
    return res_wav

def src_alignment(src_wav, dst_wav,  hop_length_sec=0.005):
    ############ 無音消去
    #src のほうがdstより短くなるようにする。
    src = librosa_remixing(src_wav, top_db=35)
    dst = librosa_remixing(dst_wav, top_db=40)

    ############ 音声分解
    #harvest
    src_f0, src_t = pyworld.harvest(src, fs=fs)
    dst_f0, dst_t = pyworld.harvest(dst, fs=fs)

    #sp and ap
    src_sp = pyworld.cheaptrick(src, src_f0, src_t, fs=fs)
    dst_sp = pyworld.cheaptrick(dst, dst_f0, dst_t, fs=fs)
    src_ap = pyworld.d4c(src, src_f0, src_t, fs=fs)
    dst_ap = pyworld.d4c(dst, dst_f0, dst_t, fs=fs)
    src_mel = librosa.feature.melspectrogram(src, n_fft=pyworld.get_cheaptrick_fft_size(fs, 71.0),
            hop_length=librosa.time_to_samples(hop_length_sec,  sr), sr=sr).T
    dst_mel = librosa.feature.melspectrogram(dst, n_fft=pyworld.get_cheaptrick_fft_size(fs, 71.0),
            hop_length=librosa.time_to_samples(hop_length_sec,  sr), sr=sr).T

    #先に正規化するとこいつがアウト！ >> 正規化は最後で！
    #sp2mcが失敗するとdtwでつまる。
    # src_sp_coded = pysptk.sp2mc(src_sp, order=32, alpha=pysptk.util.mcepalpha(fs))
    # src_sp_coded = trim_zeros_frames(src_sp_coded)
    # dst_sp_coded = pysptk.sp2mc(dst_sp, order=32, alpha=pysptk.util.mcepalpha(fs))
    # dst_sp_coded = trim_zeros_frames(dst_sp_coded)

    ###########アライメント ### melspectrum だけ！
    try:
        src_mel = normalize2(src_mel)
        dst_mel = normalize2(dst_mel)
        res = dtw(src_mel, dst_mel, step_pattern="typeIb")
        path = res.get_warping_path(target='query')
        # src_f0 = src_f0[path]
        # src_sp = src_sp[path, :]
        # src_ap = src_ap[path, :]
        # src_mel = src_mel[path, :]
        src_mel = src_mel[path, :]
    except:
        print('Error Happened!')
        if len(dst_wav) > len(src_wav):
            dst_wav = dst_wav[int(sr * 0.2): -int(sr * 0.2)]
        else:
            src_wav = src_wav[int(sr * 0.2): -int(sr * 0.2)]
        return src_alignment(src_wav, dst_wav)

    #Test!
    # wave_play(pyworld.synthesize(src_f0, src_sp, src_ap, fs))
    # wave_play(pyworld.synthesize(dst_f0, dst_sp, dst_ap, fs))
    ########　正規化
    src_f0 = (src_f0) / 800.0
    dst_f0 = (dst_f0) / 800.0
    src_sp = normalize2(src_sp)
    dst_sp = normalize2(dst_sp)
    src_ap = normalize2(src_ap)
    dst_ap = normalize2(dst_ap)

    return src_f0, dst_f0, src_sp, dst_sp, src_ap, dst_ap, src_mel, dst_mel

def _data_padding(data, length, mode):
    if mode == 'f0':
        data = np.pad(data, [0, length - len(data)], 'constant')
    elif mode in ['sp', 'ap', 'mel', 'mel2sp']:
        data = np.pad(data, ((0, length - len(data)), (0, 0)), 'constant')
    else:
        raise ValueError('Data padding mode must be f0, sp or ap')
    return data

def normalize2(wav):
    return (wav-np.min(wav))/(np.max(wav) - np.min(wav))

def parse_f0_features(data_type,  dst_who, src_f0, tgt_f0):
    assert src_f0.shape == tgt_f0.shape
    data = tf.train.Example(features=tf.train.Features(feature={
        'data_type' :_bytes_feature(data_type.encode('utf-8')),
        'dst_who' : _bytes_feature(dst_who.encode('utf-8')),
        'src_f0': _float_feature(src_f0),
        'dst_f0' : _float_feature(tgt_f0),
    }))
    return data.SerializeToString()

def parse_sp_features(data_type, dst_who, src_sp, tgt_sp):
    assert src_sp.shape == tgt_sp.shape
    data = tf.train.Example(features=tf.train.Features(feature={
        'data_type' :_bytes_feature(data_type.encode('utf-8')),
        'dst_who' : _bytes_feature(dst_who.encode('utf-8')),
        'src_sp': _float_feature(src_sp),
        'dst_sp' : _float_feature(tgt_sp),
    }))
    return data.SerializeToString()

def parse_ap_features(data_type, dst_who, src_ap, tgt_ap):
    assert src_ap.shape == tgt_ap.shape
    data = tf.train.Example(features=tf.train.Features(feature={
        'data_type' :_bytes_feature(data_type.encode('utf-8')),
        'dst_who' : _bytes_feature(dst_who.encode('utf-8')),
        'src_ap': _float_feature(src_ap),
        'dst_ap' : _float_feature(tgt_ap),
    }))
    return data.SerializeToString()

def parse_mel_features(data_type, dst_who, src_mel, tgt_mel):
    assert src_mel.shape == tgt_mel.shape
    data = tf.train.Example(features=tf.train.Features(feature={
        'data_type' :_bytes_feature(data_type.encode('utf-8')),
        'dst_who' : _bytes_feature(dst_who.encode('utf-8')),
        'src_mel': _float_feature(src_mel),
        'dst_mel' : _float_feature(tgt_mel),
    }))
    return data.SerializeToString()

def parse_mel2sp_features(data_type, dst_who, tgt_mel, tgt_sp):
    assert tgt_mel.shape[0] == tgt_sp.shape[0]
    data = tf.train.Example(features=tf.train.Features(feature={
        'data_type' :_bytes_feature(data_type.encode('utf-8')),
        'dst_who' : _bytes_feature(dst_who.encode('utf-8')),
        'dst_mel': _float_feature(tgt_mel),
        'dst_sp' : _float_feature(tgt_sp),
    }))
    return data.SerializeToString()

def get_pair_paths(dst_numbers):
    return get_nonpara_paths(dst_numbers) + get_parallel_paths(dst_numbers)

def get_parallel_paths(dst_numbers):
    res_list = list()
    src_path = './DataStore2/parallel100/127_parallel100'
    data_type = 'parallel'
    for dst_num in dst_numbers:
        src_files = os.listdir(src_path)
        dst_files = [dst_num + '_' + file.split('_')[-1] for file in src_files]
        dst_path = './DataStore2/parallel100/' + str(dst_num) +'_parallel100'
        assert [file.split('_')[-1] for file in src_files] == [file.split('_')[-1] for file in dst_files]
        for src_file, dst_file in zip(src_files, dst_files):
            src_file_path = os.path.join(src_path, src_file)
            dst_file_path = os.path.join(dst_path, dst_file)
            res_list.append((data_type, dst_num, src_file_path, dst_file_path))
    return res_list

def get_nonpara_paths(dst_numbers):
    res_list = list()
    data_type = 'nonpara'
    for dst_num in dst_numbers:
        src_path = './DataStore2/nonpara30/' + dst_num + '/' + '127' + '_nonpara30'
        dst_path = './DataStore2/nonpara30/' + dst_num + '/' + dst_num + '_nonpara30'
        src_files = os.listdir(src_path)
        dst_files = os.listdir(dst_path)
        assert [file.split('_')[-1] for file in src_files] == [file.split('_')[-1] for file in dst_files]
        for src_file, dst_file in zip(src_files, dst_files):
            src_file_path = os.path.join(src_path, src_file)
            dst_file_path = os.path.join(dst_path, dst_file)
            res_list.append((data_type, dst_num, src_file_path, dst_file_path))
    return res_list


def parse_features(data_type, dst_who, src, dst, mode):
    assert mode in ['f0', 'sp', 'ap', 'mel', 'mel2sp']
    if mode == 'f0':
        return parse_f0_features(data_type, dst_who, src, dst)
    elif mode == 'sp':
        return parse_sp_features(data_type, dst_who, src, dst)
    elif mode == 'mel':
        return parse_mel_features(data_type, dst_who, src, dst)
    elif mode == 'mel2sp':
        return parse_mel2sp_features(data_type, dst_who, src, dst)
    else:
        return parse_ap_features(data_type, dst_who, src, dst)

def data_padding(src_f0, dst_f0, src_sp, dst_sp, src_ap, dst_ap, src_mel, dst_mel, length, mode):
    assert mode in ['f0', 'sp', 'ap', 'mel', 'mel2sp']
    if mode == 'f0':
        src, dst = _data_padding(src_f0, length, mode), _data_padding(dst_f0, length, mode)
    elif mode == 'sp':
        src, dst = _data_padding(src_sp, length, mode), _data_padding(dst_sp, length, mode)
    elif mode == 'mel':
        src, dst = _data_padding(src_mel, length, mode), _data_padding(dst_mel, length, mode)
    elif mode == 'mel2sp':
        src, dst = _data_padding(dst_mel, length, mode), _data_padding(dst_sp, length, mode)
    else:
        src, dst = _data_padding(src_ap, length, mode), _data_padding(dst_ap, length, mode)
    return src, dst

def main(mode, base_length = 128):
    paths = get_pair_paths(['084', '093'])
    tf_base_path = './DataStore2/TFRecords'
    for i in trange(len(paths)):
        data_type, dst_who, src_path, dst_path = paths[i]
        print(paths[i])
        src_wav, dst_wav = librosa.load(src_path, sr=sr)[0], librosa.load(dst_path, sr=sr)[0]
        src_f0, dst_f0, src_sp, dst_sp, src_ap, dst_ap, src_mel, dst_mel= src_alignment(src_wav, dst_wav)
        length = (max(len(src_f0), len(dst_f0)) // base_length + 1) * base_length

        #Data Padding (Data is regularized, so this is 0 padding.)
        src, dst = data_padding(src_f0, dst_f0, src_sp, dst_sp, src_ap, dst_ap, src_mel, dst_mel, length, mode)
        
        #Tfrecording
        num = src_path.split('.')[-2].split('_')[-1]
        tfpath = tf_base_path + '/' + data_type + '_'+ dst_who + '_' + num + '_' + mode + '.tfrecords2'
        print(tfpath + 'is Record path') #Record must be List Form!
        with tf.io.TFRecordWriter(tfpath) as writer:
            for i in range(len(src) // base_length + 1):
                src_clip = src[i * base_length: (i+1)*base_length]
                dst_clip = dst[i * base_length: (i+1)*base_length]
                writer.write(parse_features(data_type, dst_who, src_clip, dst_clip, mode))
            for j in range(5):
                start = np.random.randint(1, base_length-1)
                src_min = src[start: -(base_length-start)]
                dst_min = dst[start: -(base_length-start)]
                for i in range(len(src_min) // base_length + 1):
                    src_clip = src_min[i * base_length: (i+1)*base_length]
                    dst_clip = dst_min[i * base_length: (i+1)*base_length]
                    writer.write(parse_features(data_type, dst_who, src_clip, dst_clip, mode))

if __name__=='__main__':
    main('mel2sp')