fs = 24000
from tqdm import trange
from NoUse.test_voice_data_augmentation2 import *
import os



def data_loader(experiment_folder):
    files = os.listdir(experiment_folder)
    source_wav_list = list()
    target_wav_list = list()
    for i in trange(len(files)):
        file = files[i]
        file_path = os.path.join(experiment_folder, file)
        rate, wav = wavfile.read(file_path)
        wav = wav.astype(np.float64)
        if file.startswith('093') or file.startswith('084'):
            target_wav_list.append(wav)
        else:
            source_wav_list.append(wav)
    res_list = list()
    for src in source_wav_list:
        for tgt in target_wav_list:
            res_list.append((src, tgt))
    return res_list


def split_factors(data, length = 513, stride = 256, mode='f0'):
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
def preprocess(wav_list, length=513, frame_period=5, log_nml=False):
    f0_list_full, f0_list, sp_list, ap_list = list(), list(), list(), list()
    for i in trange(len(wav_list)):
        src_wav, tgt_wav = wav_list[i]

        #Alignment済みのwavファイルの長さは同じ
        src_f0, src_t,  src_sp, src_ap = world_decompose(src_wav, fs, frame_period=frame_period)
        tgt_f0, tgt_t,  tgt_sp, tgt_ap = world_decompose(tgt_wav, fs, frame_period=frame_period)
        src_f0 = src_f0 / 800.0 # Maximum f0 level in world harvest
        tgt_f0 = tgt_f0 / 800.0
        if log_nml:
            src_sp = log_normalize(src_sp)
            src_ap = log_normalize(src_ap)
            tgt_sp = log_normalize(tgt_sp)
            tgt_ap = log_normalize(tgt_ap)
        else:
            src_sp = normalize(src_sp)
            src_ap = normalize(src_ap)
            tgt_sp = normalize(tgt_sp)
            tgt_ap = normalize(tgt_ap)
        if len(src_wav) != len(tgt_wav):
            min_len = min(len(src_f0), len(tgt_f0))
            src_f0, tgt_f0, src_sp, src_ap, tgt_sp, tgt_ap = \
                src_f0[:min_len], tgt_f0[:min_len], src_sp[:min_len, :], src_ap[:min_len, :], tgt_sp[:min_len, :], tgt_ap[:min_len, :]
        f0_list_full.append((src_f0, tgt_f0))
        for src, tgt in zip(split_factors(src_f0, length=length, mode='f0'), split_factors(tgt_f0, length=length, mode='f0')):
            f0_list.append((src, tgt))
        for src, tgt in zip(split_factors(src_sp, length=length , mode='sp'), split_factors(tgt_sp,length=length, mode='sp')):
            if length <= 256:
                sp_list.append((src[:, :256], tgt[:,  :256]))
            elif length == 513:
                sp_list.append((src, tgt))
            else:
                raise ValueError('length must be 513 or 256')
        for src, tgt in zip(split_factors(src_ap, length=length ,mode='ap'), split_factors(tgt_ap, length=length, mode='ap')):
            if length <= 256:
                ap_list.append((src[:, :256], tgt[:, :256]))
            elif length == 513:
                ap_list.append((src, tgt))
            else:
                raise ValueError('length must be 513 or 256')
    return (f0_list_full, f0_list, sp_list, ap_list)


def load_wav(file_path):
    rate, wav = wavfile.read(file_path)
    wav = wav.astype(np.float64)
    return wav

def save_npz(save_folder_path, file_name, full_f0, f0, sp, ap):
    path = os.path.join(save_folder_path, file_name + '.npz')
    np.savez(path, full_f0=full_f0, f0=f0, sp=sp, ap=ap)

def load_npz(save_folder_path, file_name):
    path = os.path.join(save_folder_path, file_name + '.npz')
    return np.load(path)

def librosa_remixing(wav, top_db):
    wav = wav.astype(np.float64)
    res_wav = librosa.effects.remix(wav, intervals=librosa.effects.split(wav, top_db=top_db, ref=np.mean, frame_length=256,  hop_length=256))
    return res_wav
import librosa


def test_data_maker(data_folder, save_folder, save_name, frame_period= 5, length=513, log_nml=False):
    wav_files = os.listdir(data_folder)
    f0_list_full, f0_list, sp_list, ap_list = list(), list(), list(), list()
    for file in wav_files:
        wav = load_wav(os.path.join(data_folder, file))
        wav = librosa_remixing(wav, 50)
        f0, t,  sp, ap = world_decompose(wav, fs, frame_period)
        f0 = (f0-400) / 800.0 # Maximum f0 level in world harvest
        if log_nml:
            sp = log_normalize(sp)
            ap = log_normalize(ap)
        else:
            sp = normalize(sp)
            ap = normalize(ap)
        f0_list_full.append(f0)
        for split_f0 in split_factors(f0, length=length, mode='f0'):
            f0_list.append(split_f0)
        for split_sp in split_factors(sp, length=length,  mode='sp'):
            if length <= 256:
                sp_list.append(split_sp[:, :256])
            else:
                sp_list.append(split_sp)
        for split_ap in split_factors(ap , length=length,  mode='ap'):
            if length <= 256:
                ap_list.append(split_ap[:, :256])
            else:
                ap_list.append(split_ap)
    f0_list = np.array(f0_list)
    sp_list = np.array(sp_list)
    ap_list = np.array(ap_list)
    save_npz(save_folder, save_name, f0_list_full, f0_list, sp_list, ap_list)

if __name__=='__main__':
    #Test Data Making Test
    # experiment_folder = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/093_aligned_experiment'
    # # experiment_folder = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/zero_cut_experiment'
    # wav_list = data_loader(experiment_folder)
    save_folder = './DataStore2'
    # valid_save_folder = './DataStoreValidation'

    ############ Data Augment processing! ###########################
    # res_dict = data_augmentations(wav_list)
    # for name, wavs in res_dict.items():
    #     print('preprocess {} is starting...'.format(name))
    #     full_f0, f0, sp, ap = preprocess(wav_list)
    #     save_npz(save_folder, name , full_f0, f0, sp, ap)

    # ############## training dataの作成
    # full_f0, f0, sp, ap = preprocess(wav_list, frame_period=5, length=256, log_nml=True) #Already normalized...
    # save_npz(save_folder, 'only_normalized_256_256_log', full_f0, f0, sp, ap)

    ############## test dataの作成　#################################
    test_wav_folder = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/my_voices/without_alignment'
    test_data_maker(test_wav_folder, save_folder, 'valid_without_alignment_256_256', length=256)
    test_wav_folder = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/my_voices/alignment'
    test_data_maker(test_wav_folder, save_folder, 'valid_alignment_256_256', length=256)