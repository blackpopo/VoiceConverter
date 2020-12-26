fs = 24000
from utilities import  *
from DataStore.data_arrangement import *
import tensorflow as tf


def datamaker(base_source_folder,  target_num, start=0, end=-1):
    npz_flies = os.listdir(base_source_folder)
    total_src_f0, total_tgt_f0, total_src_sp, total_tgt_sp, total_src_sp_full, total_tgt_sp_full =\
        list(), list(), list(), list(), list(), list()
    for npz in tqdm(npz_flies[start: end]):
        print('file {} is starting'.format(npz) )
        data = np.load(os.path.join(base_source_folder, npz), allow_pickle=True)
        dst = data["target"]
        who, dst_f0, dst_sp = dst
        source_data = data["source"]
        assert target_num == who
        src_f0_list, tgt_f0_list, src_sp_list, tgt_sp_list, src_sp_list_full, tgt_sp_list_full = processor(dst_f0, dst_sp, source_data)
        total_src_f0 = total_src_f0 + src_f0_list
        total_tgt_f0 = total_tgt_f0 + tgt_f0_list
        total_src_sp = total_src_sp + src_sp_list
        total_tgt_sp = total_tgt_sp + tgt_sp_list
        total_src_sp_full = total_src_sp_full + src_sp_list_full
        total_tgt_sp_full = total_tgt_sp_full + tgt_sp_list_full
        assert len(total_tgt_f0) == len(total_src_sp) == len(total_tgt_f0) == len(total_src_f0)
    total_src_f0, total_tgt_f0, total_src_sp, total_tgt_sp, total_src_sp_full, total_tgt_sp_full =\
        np.array(total_src_f0, dtype=np.float32), np.array(total_tgt_f0, dtype=np.float32), np.array(total_src_sp, dtype=np.float32), np.array(total_tgt_sp, np.float32) , np.array(total_src_sp_full, dtype=np.float32), np.array(total_tgt_sp_full, np.float32)
    return total_src_f0, total_tgt_f0, total_src_sp, total_tgt_sp, total_src_sp_full, total_tgt_sp_full

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

def processor(dst_f0, dst_sp , source_wav):
    dst_f0_list, dst_sp_list, dst_sp_full_list = list(), list(), list()
    total_src_f0, total_tgt_f0, total_src_sp, total_tgt_sp, total_src_sp_full, total_tgt_sp_full =\
        list(), list(), list(), list(), list(), list()
    for src in split_factors(dst_f0, mode='f0'):
        dst_f0_list.append(src)
    for src in split_factors(dst_sp, mode='sp'):
        dst_sp_list.append(src[:, :256])
    for src in split_factors(dst_sp, mode='sp'):
        dst_sp_full_list.append(src)
    for who, src_f0, src_sp in source_wav:
        print('name {} is starting'.format(who))
        src_f0_list, tgt_f0_list, src_sp_list, tgt_sp_list, src_sp_list_full, tgt_sp_list_full = preprocess(src_f0, src_sp, dst_f0_list, dst_sp_list)
        total_src_f0 = total_src_f0 + src_f0_list
        total_tgt_f0 = total_tgt_f0 + tgt_f0_list
        total_src_sp = total_src_sp + src_sp_list
        total_tgt_sp = total_tgt_sp + tgt_sp_list
        total_src_sp_full = total_src_sp_full + src_sp_list_full
        total_tgt_sp_full = total_tgt_sp_full + tgt_sp_list_full
    return total_src_f0, total_tgt_f0, total_src_sp, total_tgt_f0, total_src_sp_full, total_tgt_sp_full

#ap, sp, f0を抜き出し+zero_paddingをしてから、sp, ap, f0の抜き出し
#いつ正規化しよう？データを保存する前？あと？　各wav　それとも全体？　>> 前かつ各wav
def preprocess(f0, sp, tgt_f0_list, tgt_sp_list, tgt_full_list, log_nml=False):
    f0_list, sp_list, sp_full_list = list(), list(), list()
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
    for src in split_factors(sp, mode='sp'):
        sp_full_list.append(src)
    if len(tgt_f0_list) != len(f0_list):
        min_len = min(len(tgt_f0_list), len(f0_list))
        f0_list, tgt_f0_list, sp_list, tgt_sp_list, sp_full_list, tgt_full_list =\
            f0_list[:min_len, :], tgt_f0_list[:min_len, :, :], sp_list[:min_len, :, :], tgt_sp_list[:min_len, :, :],  sp_full_list[:min_len, :, :], tgt_full_list[:min_len, :, :]
    return f0_list, tgt_f0_list, sp_list, tgt_sp_list, sp_full_list, tgt_full_list

def datamaker2(target_number, save_path, start=0, end=-1):
    source_base_folder = "C:/Users/Atsuya/Documents/Sites/Voice/jvs_ver1/jvs_ver1"
    folders = os.listdir(source_base_folder)
    paths = collect_same_num_path(source_base_folder, folders)
    save_path = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/ProductionModel/DataStore/093_084_alignment2'
    total_src_f0, total_tgt_f0, total_src_sp, total_tgt_sp = list(), list(), list(), list()
    for wav_num, data_list in tqdm(paths.items()[start: end]):
        dst_f0, dst_sp, src_aligned = collect_wavs(target_number, wav_num, data_list, save_path)
        for who, src_f0, src_sp in src_aligned:
            src_f0_list, tgt_f0_list, src_sp_list, tgt_sp_list = processor(dst_f0, dst_sp, src_sp)
            total_src_f0 = total_src_f0 + src_f0_list
            total_tgt_f0 = total_tgt_f0 + tgt_f0_list
            total_src_sp = total_src_sp + src_sp_list
            total_tgt_sp = total_tgt_sp + tgt_sp_list
    np.savez(os.path.join(save_path, target_number + '_' + str(start) + '_' + str(end)), src_f0=total_src_f0,
         tgt_f0=total_tgt_f0, src_sp=total_src_sp, tgt_sp=total_tgt_sp)


def _float_feature(value):
  """float / double 型から float_list を返す"""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """bool / enum / int / uint 型から Int64_list を返す"""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def main():
    #datamaker1 version
    target_nums = ["093", "084"]
    save_folder =  'C:/Users/Atsuya/PycharmProjects/VoiceConverter/ProductionModel/DataStore/' + '_'.join(target_nums) +'_alignment2'
    total_src_f0, total_tgt_f0, total_src_sp, total_tgt_sp, total_src_sp_full, total_tgt_sp_full =\
        list(), list(), list(), list(), list(), list()
    for target_num in target_nums:
        source_folder = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/' + target_num + '_aligned2'
        src_f0, tgt_f0, src_sp, tgt_sp, src_sp_full, tgt_sp_full= datamaker(source_folder, target_num)
        total_src_f0 = total_src_f0 + src_f0
        total_tgt_f0 = total_tgt_f0 + tgt_f0
        total_src_sp = total_src_sp + src_sp
        total_tgt_sp = total_tgt_sp + tgt_sp
        total_src_sp_full = total_src_sp_full + src_sp_full
        total_tgt_sp_full = total_tgt_sp_full + tgt_sp_full

if __name__=='__main__':
    main()