fs = 24000
from tqdm import trange
from utilities import  *
import os
from tqdm import tqdm
import sys
import tensorflow as tf
import glob

def datamaker(save_path, base_source_folder,  target_num, mode):
    npz_flies = os.listdir(base_source_folder)
    saved_files = glob.glob(save_path + '/*')
    nums = [file.split('.')[-2][-3:] for file in saved_files]
    dst_name = [file.split('.')[-2][-7:-4] for file in saved_files]
    print(nums)
    print(dst_name)
    for npz in tqdm(npz_flies):
        wav_num = npz.split('.')[0]
        if wav_num in nums and target_num == dst_name:
            pass
        else:
            print('file {} is starting'.format(npz) )
            data = np.load(os.path.join(base_source_folder, npz), allow_pickle=True)
            dst = data["target"]
            dst_who, dst_f0, dst_sp = dst
            source_data = data["source"]
            assert target_num == dst_who
            tfpath = os.path.join(save_path, mode + '_' + dst_who + '_' + wav_num + '.tfrecords')
            processor(tfpath, wav_num, dst_who, dst_f0, dst_sp, source_data, mode)

def split_factors(data, length = 128, stride = 128, mode='f0'):
    if not mode in ['f0', 'sp', 'ap']:
        raise ValueError('mode must be in [f0, sp, ap]')
    split_list = []
    for i in range(len(data) // stride + 1):
        if length + stride*i < len(data):
            split_list.append(data[stride * i: length + stride* i])
        else:
            res_data = data_padding(data[stride* i:], length, mode)
            split_list.append(res_data)
    return split_list

def np_create_segmap(image):
    mean = np.mean(image)
    segmap = np.zeros_like(image)
    segmap[np.where(image > mean)] = 1
    return segmap

def processor(tfpath,  wav_num, dst_who, dst_f0, dst_sp , source_data, mode, log_nml=False):
    dst_f0_list, dst_sp_list, dst_sp_full_list = list(), list(), list()
    dst_f0 = (dst_f0 -400.0) / 400.0 # Maximum f0 level in world harvest
    if log_nml:
        dst_sp = log_normalize(dst_sp)
    else:
        dst_sp = normalize(dst_sp)
    for src in split_factors(dst_f0, mode='f0'):
        dst_f0_list.append(src)
    for src in split_factors(dst_sp, mode='sp'):
        dst_sp_list.append(src[:, :256])
    for src in split_factors(dst_sp, mode='sp'):
        dst_sp_full_list.append(src)
    with tf.io.TFRecordWriter(tfpath) as writer:
        #1つのwav_numの形式で書き込む
        for src_who, src_f0, src_sp in source_data:
            print('name {} is starting'.format(src_who))
            src_f0_list, tgt_f0_list, src_sp_list, tgt_sp_list, tgt_sp_full_list = preprocess(src_f0, src_sp, dst_f0_list, dst_sp_list, dst_sp_full_list)
            for i, src_f0, tgt_f0, src_sp, tgt_sp,  tgt_sp_full in \
                    zip(range(1, len(src_f0_list)+1), src_f0_list, tgt_f0_list, src_sp_list, tgt_sp_list, tgt_sp_full_list):
                if mode == 'f0':
                    rec = parse_f0_features(wav_num, src_who, dst_who, src_f0, tgt_f0)
                elif mode == 'sp_spade':
                    rec = parse_sp_spade_features(wav_num, src_who, dst_who, src_sp, tgt_sp_full_list)
                elif mode == 'sp':
                    rec = parse_sp_features(wav_num, src_who, dst_who, src_sp, tgt_sp)
                else:
                    raise ValueError
                writer.write(rec)

#ap, sp, f0を抜き出し+zero_paddingをしてから、sp, ap, f0の抜き出し
#いつ正規化しよう？データを保存する前？あと？　各wav　それとも全体？　>> 前かつ各wav
def preprocess(f0, sp, tgt_f0_list, tgt_sp_list, tgt_full_list, log_nml=False):
    f0_list, sp_list = list(), list()
    #Alignment済みのwavファイルの長さは同じ)
    f0 = (f0 -400.0) / 400.0 # Maximum f0 level in world harvest
    if log_nml:
        sp = log_normalize(sp)
    else:
        sp = normalize(sp)
    for src in split_factors(f0, mode='f0'):
        f0_list.append(src)
    for src in split_factors(sp, mode='sp'):
        sp_list.append(src[:, :256])
    if len(tgt_f0_list) != len(f0_list):
        min_len = min(len(tgt_f0_list), len(f0_list))
        f0_list, tgt_f0_list, sp_list, tgt_sp_list,  tgt_full_list =\
            f0_list[:min_len, :], tgt_f0_list[:min_len, :, :], sp_list[:min_len, :, :], tgt_sp_list[:min_len, :, :], tgt_full_list[:min_len, :, :]
    return f0_list, tgt_f0_list, sp_list, tgt_sp_list,  tgt_full_list

# def datamaker2(target_number, save_path, start=0, end=-1):
#     source_base_folder = "C:/Users/Atsuya/Documents/Sites/Voice/jvs_ver1/jvs_ver1"
#     folders = os.listdir(source_base_folder)
#     paths = collect_same_num_path(source_base_folder, folders)
#     save_path = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/ProductionModel/DataStore/093_084_alignment2'
#     for wav_num, data_list in tqdm(paths.items()[start: end]):
#         dst_f0, dst_sp, src_aligned = collect_wavs(target_number, wav_num, data_list, save_path)
#         processor(wav_num, target_number, dst_f0, dst_sp , src_aligned)

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

def parse_f0_features(wav_num, src_who, dst_who, src_f0, tgt_f0):
    data = tf.train.Example(features=tf.train.Features(feature={
        'wav_num' :_bytes_feature(wav_num.encode('utf-8')),
        'src_who' : _bytes_feature(src_who.encode('utf-8')),
        'dst_who' : _bytes_feature(dst_who.encode('utf-8')),
        'src_f0': _float_feature(src_f0),
        'tgt_f0' : _float_feature(tgt_f0),
    }))
    return data.SerializeToString()

def parse_sp_features(wav_num, src_who, dst_who, src_sp, tgt_sp):
    segmaps = list()
    for sp in tgt_sp:
        segmaps.append(np_create_segmap(sp))
    data = tf.train.Example(features=tf.train.Features(feature={
        'wav_num' :_bytes_feature(wav_num.encode('utf-8')),
        'src_who' : _bytes_feature(src_who.encode('utf-8')),
        'dst_who' : _bytes_feature(dst_who.encode('utf-8')),
        'src_sp': _float_feature(src_sp),
        'tgt_sp' : _float_feature(tgt_sp),
        'segmap' : _float_feature(np.array(segmaps))
    }))
    return data.SerializeToString()

def parse_sp_spade_features(wav_num, src_who, dst_who, src_sp, tgt_full_sp):
    segmaps = list()
    for sp in tgt_full_sp:
        segmaps.append(np_create_segmap(sp))
    tgt_full_sp = np.array(tgt_full_sp, dtype=np.float32)
    segmaps = np.array(segmaps, dtype=np.float32)
    data = tf.train.Example(features=tf.train.Features(feature={
        'wav_num': _bytes_feature(wav_num.encode('utf-8')),
        'src_who': _bytes_feature(src_who.encode('utf-8')),
        'dst_who': _bytes_feature(dst_who.encode('utf-8')),
        'src_sp' : _float_feature(src_sp),
        'tgt_sp_full': _float_feature(tgt_full_sp),
        'segmap_full': _float_feature(segmaps)
    }))
    return data.SerializeToString()


def main():
    #datamaker1 version
    target_nums = ["093", "084"]
    mode = 'sp'
    save_folder =  'C:/Users/Atsuya/PycharmProjects/VoiceConverter/ProductionModel/DataStore/' + '_'.join(target_nums) + '_' + mode
    for target_num in target_nums:
        source_folder = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/' + target_num + '_aligned2'
        datamaker(save_folder, source_folder,  target_num, mode)


if __name__=='__main__':
    main()
    # renaming()