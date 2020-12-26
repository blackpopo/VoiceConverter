fs = 24000
from tqdm import trange
from utilities import  *
import os
from scipy.io import wavfile

def file_arrange(base_source_folder):
    folders = os.listdir(base_source_folder)
    temp_list = list()
    for folder in folders:
        files = os.listdir(os.path.join(base_source_folder, folder))
        for file in files:
            who = file[0:3]
            to = file[4:7]
            num = file.split('.')[0][-3:]
            path = os.path.join(base_source_folder, folder, file)
            temp_list.append((num, who, path, to))
    path_list = list()
    for s_num, s_who, s_path, s_to in temp_list:
        for t_num, t_who, t_path, t_to in temp_list:
            if (t_who == '093' or t_who == '084') and s_num == t_num and t_who == t_to:
                path_list.append((s_path, t_path))
    return path_list

def data_maker():
    base_path1 = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/084_aligned2'
    base_path2 = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/093_aligned2'
    save_folder = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/ProductionModel/DataStore/093_084_alignment2'
    file_name = '093_084_aligned2'
    save_file_name = 'saved_files2.txt'
    with open(os.path.join(save_folder, save_file_name), 'r') as f:
        saved_paths = f.readlines()
    saved_paths = [f.rstrip('\n') for f in saved_paths]
    paths = file_arrange(base_path1) + file_arrange(base_path2)
    paths = [(src, tgt) for src, tgt in paths if src+'\t'+tgt not in saved_paths]
    window = len(paths) // 10
    for i in trange(len(paths) // window + 1):
        total_src_f0, total_tgt_f0, total_src_sp, total_tgt_sp = list(), list(), list(), list()
        start = i* window
        if (i+1) * window > len(paths):
            end = len(paths)
        else:
            end = (i+1) * window
        temp_paths = paths[start: end]
        for j in trange(len(temp_paths)):
            src_path, tgt_path  = paths[j]
            print('Src: {}  Tgt: {}'.format(src_path, tgt_path))
            src_f0_list, tgt_f0_list, src_sp_list, tgt_sp_list = preprocess(src_path, tgt_path)
            total_src_f0 = total_src_f0 + src_f0_list
            total_tgt_f0 = total_tgt_f0 + tgt_f0_list
            total_src_sp = total_src_sp + src_sp_list
            total_tgt_sp = total_tgt_sp + tgt_sp_list
            saved_paths.append(src_path + '\t' + tgt_path)

        save_npz(os.path.join(save_folder), file_name + '_ds' + str(i) , total_src_f0, total_tgt_f0, total_tgt_sp, total_tgt_sp)
        with open(os.path.join(save_folder, save_file_name), 'w') as f:
            f.write('\n'.join(saved_paths))


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

#ap, sp, f0を抜き出し+zero_paddingをしてから、sp, ap, f0の抜き出し
#いつ正規化しよう？データを保存する前？あと？　各wav　それとも全体？　>> 前かつ各wav
def preprocess(src_path, tgt_path, log_nml=False):
    src_f0_list, tgt_f0_list, src_sp_list, tgt_sp_list = list(), list(), list(), list()
    src_wav, tgt_wav = np.load(src_path).astype(np.float64), np.load(tgt_path).astype(np.float64)
    #Alignment済みのwavファイルの長さは同じ
    src_f0, src_t,  src_sp, src_ap = world_decompose(src_wav, fs)
    tgt_f0, tgt_t,  tgt_sp, tgt_ap = world_decompose(tgt_wav, fs)
    src_f0 = (src_f0 -400.0) / 400.0 # Maximum f0 level in world harvest
    tgt_f0 = (tgt_f0 -400.0) / 400.0
    if log_nml:
        src_sp = log_normalize(src_sp)
        tgt_sp = log_normalize(tgt_sp)
    else:
        src_sp = normalize(src_sp)
        tgt_sp = normalize(tgt_sp)
    if len(src_wav) != len(tgt_wav):
        min_len = min(len(src_f0), len(tgt_f0))
        src_f0, tgt_f0, src_sp, src_ap, tgt_sp, tgt_ap = \
            src_f0[:min_len], tgt_f0[:min_len], src_sp[:min_len, :], src_ap[:min_len, :], tgt_sp[:min_len, :], tgt_ap[:min_len, :]
    for src, tgt in zip(split_factors(src_f0, mode='f0'), split_factors(tgt_f0,mode='f0')):
        src_f0_list.append(src)
        tgt_f0_list.append(tgt)
    for src, tgt in zip(split_factors(src_sp, mode='sp'), split_factors(tgt_sp, mode='sp')):
        src_sp_list.append(src[:, :256])
        tgt_sp_list.append(tgt[:, :256])
    return src_f0_list, tgt_f0_list, src_sp_list, tgt_sp_list


def load_wav(file_path):
    rate, wav = wavfile.read(file_path)
    wav = wav.astype(np.float64)
    return wav

def save_npz(save_folder_path, file_name, src_f0, tgt_f0, src_sp=None, tgt_sp=None):
    path = os.path.join(save_folder_path, file_name + '.npz')
    if src_sp is  None and tgt_sp is  None:
        np.savez(path, f0=src_f0, sp=tgt_f0)
    else:
        np.savez(path, src_f0=src_f0, tgt_f0=tgt_f0, src_sp=src_sp, tgt_sp=tgt_sp)


def load_npz(save_folder_path, file_name):
    path = os.path.join(save_folder_path, file_name + '.npz')
    return np.load(path)

def test_data_maker(data_folder, save_folder, save_name, log_nml=False):
    wav_files = os.listdir(data_folder)
    f0_list, sp_list  = list(), list()
    for file in wav_files:
        wav = load_wav(os.path.join(data_folder, file))
        f0, t,  sp, ap = world_decompose(wav, fs)
        f0 = (f0-800.0) / 800.0 # Maximum f0 level in world harvest
        if log_nml:
            sp = log_normalize(sp)
        else:
            sp = normalize(sp)
        for split_f0 in split_factors(f0, mode='f0'):
            f0_list.append(split_f0)
        for split_sp in split_factors(sp,  mode='sp'):
            sp_list.append(split_sp[:, :256])
    f0_list = np.array(f0_list)
    sp_list = np.array(sp_list)
    save_npz(save_folder, save_name,  f0_list, sp_list)

if __name__=='__main__':
    data_maker()
