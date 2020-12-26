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

def src_alignment(src_wav, dst_wav,  mode, hop_length_sec=0.005):
    src_f0, dst_f0, src_sp, dst_sp, src_ap, dst_ap, src_mel, dst_mel =\
        list(), list(), list(), list(), list(), list(), list(), list()
    assert mode in ['f0', 'sp', 'ap', 'mel']
    ############ 無音消去
    #src のほうがdstより短くなるようにする。
    src = librosa_remixing(src_wav, top_db=40)
    dst = librosa_remixing(dst_wav, top_db=40)

    ############ 音声分解

    #harvest
    if mode in ['f0', 'sp', 'ap']:
        src_f0, src_t = pyworld.harvest(src, fs=fs)
        dst_f0, dst_t = pyworld.harvest(dst, fs=fs)

        if mode == 'f0':
            try:
                src_f0_normalized = src_f0 / 800.0
                dst_f0_normalized = dst_f0 / 800.0
                res = dtw(src_f0_normalized, dst_f0_normalized, step_pattern="typeIb")
                path = res.get_warping_path(target='query')
                src_f0 = src_f0_normalized[path]
                dst_f0 = dst_f0_normalized
            except:
                print('Error Happened!')
                if len(dst_wav) > len(src_wav):
                    dst_wav = dst_wav[int(sr * 0.2): -int(sr * 0.2)]
                else:
                    src_wav = src_wav[int(sr * 0.2): -int(sr * 0.2)]
                return src_alignment(src_wav, dst_wav, mode)

        #sp and ap
        if mode =='ap':
            src_ap = pyworld.d4c(src, src_f0, src_t, fs=fs)
            dst_ap = pyworld.d4c(dst, dst_f0, dst_t, fs=fs)
            src_ap = normalize2(src_ap)
            dst_ap = normalize2(dst_ap)

        if mode == 'sp':
            src_sp = pyworld.cheaptrick(src, src_f0, src_t, fs=fs)
            dst_sp = pyworld.cheaptrick(dst, dst_f0, dst_t, fs=fs)
            try:
                src_sp_normalized = normalize2(src_sp)
                dst_sp_normalized = normalize2(dst_sp)
                res = dtw(src_sp_normalized, dst_sp_normalized, step_pattern="typeIb")
                path = res.get_warping_path(target='query')
                src_sp = src_sp_normalized[path, :]
                dst_sp = dst_sp_normalized
            except:
                print('Error Happened!')
                if len(dst_wav) > len(src_wav):
                    dst_wav = dst_wav[int(sr * 0.2): -int(sr * 0.2)]
                else:
                    src_wav = src_wav[int(sr * 0.2): -int(sr * 0.2)]
                return src_alignment(src_wav, dst_wav, mode)

    if mode == 'mel':
        src_mel = librosa.feature.melspectrogram(src, n_fft=pyworld.get_cheaptrick_fft_size(fs, 71.0),
                hop_length=librosa.time_to_samples(hop_length_sec,  sr), sr=sr).T
        dst_mel = librosa.feature.melspectrogram(dst, n_fft=pyworld.get_cheaptrick_fft_size(fs, 71.0),
                hop_length=librosa.time_to_samples(hop_length_sec,  sr), sr=sr).T

        try:
            src_mel = normalize2(src_mel)
            dst_mel = normalize2(dst_mel)
            res = dtw(src_mel, dst_mel, step_pattern="typeIb")
            path = res.get_warping_path(target='query')
            src_mel = src_mel[path, :]
        except:
            print('Error Happened!')
            if len(dst_wav) > len(src_wav):
                dst_wav = dst_wav[int(sr * 0.2): -int(sr * 0.2)]
            else:
                src_wav = src_wav[int(sr * 0.2): -int(sr * 0.2)]
            return src_alignment(src_wav, dst_wav, mode)

    #Test!
    # wave_play(pyworld.synthesize(src_f0, src_sp, src_ap, fs))
    # wave_play(pyworld.synthesize(dst_f0, dst_sp, dst_ap, fs))
    ########　正規化
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

def parse_f0_features(src_who,  dst_who, src_f0, tgt_f0):
    assert src_f0.shape == tgt_f0.shape
    data = tf.train.Example(features=tf.train.Features(feature={
        'src_who' :_bytes_feature(src_who.encode('utf-8')),
        'dst_who' : _bytes_feature(dst_who.encode('utf-8')),
        'src_f0': _float_feature(src_f0),
        'dst_f0' : _float_feature(tgt_f0),
    }))
    return data.SerializeToString()

def parse_sp_features(src_who, dst_who, src_sp, tgt_sp):
    assert src_sp.shape == tgt_sp.shape
    data = tf.train.Example(features=tf.train.Features(feature={
        'src_who' :_bytes_feature(src_who.encode('utf-8')),
        'dst_who' : _bytes_feature(dst_who.encode('utf-8')),
        'src_sp': _float_feature(src_sp),
        'dst_sp' : _float_feature(tgt_sp),
    }))
    return data.SerializeToString()

def parse_ap_features(src_who, dst_who, src_ap, tgt_ap):
    assert src_ap.shape == tgt_ap.shape
    data = tf.train.Example(features=tf.train.Features(feature={
        'src_who' :_bytes_feature(src_who.encode('utf-8')),
        'dst_who' : _bytes_feature(dst_who.encode('utf-8')),
        'src_ap': _float_feature(src_ap),
        'dst_ap' : _float_feature(tgt_ap),
    }))
    return data.SerializeToString()

def parse_mel_features(src_who, dst_who, src_mel, tgt_mel):
    assert src_mel.shape == tgt_mel.shape
    data = tf.train.Example(features=tf.train.Features(feature={
        'src_who' :_bytes_feature(src_who.encode('utf-8')),
        'dst_who' : _bytes_feature(dst_who.encode('utf-8')),
        'src_mel': _float_feature(src_mel),
        'dst_mel' : _float_feature(tgt_mel),
    }))
    return data.SerializeToString()


def get_pair_paths(src_numbers, dst_numbers):
    res_list = list()
    base_path = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/jvs_ver1'
    data_path = 'parallel100/wav24kHz16bit'
    prefix_folder = 'jvs'
    prefix_file = 'VOICEACTRESS100_'
    for src_num in src_numbers:
        for dst_num in dst_numbers:
            for wav_num in range(1, 101):
                wav_num = str(wav_num).zfill(3)
                src_file_path = os.path.join(base_path, prefix_folder + src_num, data_path, prefix_file + wav_num + '.wav')
                dst_file_path = os.path.join(base_path, prefix_folder + dst_num, data_path, prefix_file + wav_num + '.wav')
                res_list.append( (src_num, dst_num, src_file_path, dst_file_path))
    return res_list

def parse_features(src_who, dst_who, src, dst, mode):
    assert mode in ['f0', 'sp', 'ap', 'mel', 'mel2sp']
    if mode == 'f0':
        return parse_f0_features(src_who, dst_who, src, dst)
    elif mode == 'sp':
        return parse_sp_features(src_who, dst_who, src, dst)
    elif mode == 'mel':
        return parse_mel_features(src_who, dst_who, src, dst)
    else:
        return parse_ap_features(src_who, dst_who, src, dst)

def data_padding(src_f0, dst_f0, src_sp, dst_sp, src_ap, dst_ap, src_mel, dst_mel, length, mode):
    assert mode in ['f0', 'sp', 'ap', 'mel', 'mel2sp']
    if mode == 'f0':
        src, dst = _data_padding(src_f0, length, mode), _data_padding(dst_f0, length, mode)
    elif mode == 'sp':
        src, dst = _data_padding(src_sp, length, mode), _data_padding(dst_sp, length, mode)
    elif mode == 'mel':
        src, dst = _data_padding(src_mel, length, mode), _data_padding(dst_mel, length, mode)
    else:
        src, dst = _data_padding(src_ap, length, mode), _data_padding(dst_ap, length, mode)
    return src, dst

def main(mode, base_length = 128):
    src_numbers = ['098', '032', '047', '054', '011', '097', '099', '087', '080', '075', '052', '050', '041',
                   '001', '086', '076', '070', '046', '031', '013', '005', '077''074', '073', '045', '028',
                   '020', '003', '088', '068', '023', '081', '049', '034', '033', '022', '100', '089', '079',
                   '048', '044', '037', '012', '009', '071', '078', '042', '021', '006']
    dst_numbers = ['084', '093']
    paths = get_pair_paths(src_numbers, dst_numbers)
    tf_base_path = './DataStore2/TFRecords4'
    for i in trange(len(paths)):
        src_who, dst_who, src_path, dst_path = paths[i]
        print(paths[i])
        try:
            src_wav, dst_wav = librosa.load(src_path, sr=sr)[0], librosa.load(dst_path, sr=sr)[0]
        except:
            continue
        src_f0, dst_f0, src_sp, dst_sp, src_ap, dst_ap, src_mel, dst_mel= src_alignment(src_wav, dst_wav, mode)
        if mode  == 'f0':
            length = (max(len(src_f0), len(dst_f0)) //base_length +1) * base_length
        elif mode == 'sp':
            length = (max(len(src_sp), len(dst_sp)) // base_length + 1) * base_length
        elif mode == 'ap':
            length = (max(len(src_ap), len(dst_ap)) // base_length + 1) * base_length
        else:
            length = (max(len(src_mel), len(dst_mel)) // base_length + 1) * base_length

        #Data Padding (Data is regularized, so this is 0 padding.)
        src, dst = data_padding(src_f0, dst_f0, src_sp, dst_sp, src_ap, dst_ap, src_mel, dst_mel, length, mode)

        #Tfrecording
        num = src_path.split('.')[-2].split('_')[-1]
        tfpath = tf_base_path + '/' + src_who + '_'+ dst_who + '_' + num + '_' + mode + '.tfrecords'
        print(tfpath + 'is Record path') #Record must be List Form!
        with tf.io.TFRecordWriter(tfpath) as writer:
            for i in range(len(src) // base_length):
                src_clip = src[i * base_length: (i+1)*base_length]
                dst_clip = dst[i * base_length: (i+1)*base_length]
                writer.write(parse_features(src_who, dst_who, src_clip, dst_clip, mode))
            start = np.random.randint(1, base_length-1)
            src = src[start: -(base_length-start)]
            dst = dst[start: -(base_length-start)]
            for i in range(len(src) // base_length):
                src_clip = src[i * base_length: (i+1)*base_length]
                dst_clip = dst[i * base_length: (i+1)*base_length]
                writer.write(parse_features(src_who, dst_who, src_clip, dst_clip, mode))

if __name__=='__main__':
    main('sp')