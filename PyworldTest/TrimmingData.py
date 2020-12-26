import os
from tqdm import trange, tqdm
import librosa
from nnmnkwii.preprocessing import trim_zeros_frames
import soundfile
import numpy as np
import pyworld
import pysptk
from dtwalign import dtw

def librosa_remixing(wav, top_db):
    wav = wav.astype(np.float64)
    res_wav = librosa.effects.remix(wav, intervals=librosa.effects.split(wav, top_db=top_db, ref=np.mean, frame_length=256,  hop_length=256))
    return res_wav

sr = fs = 24000

def load_wavs(source_dir, destination_dir):
    files = os.listdir(source_dir)
    res_list = []
    for i in trange(len(files)):
        file = files[i]
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(destination_dir, file)
        source_wav = librosa.core.load(src_path,  sr=sr)[0]
        destination_wav = librosa.core.load(dst_path, sr=sr)[0]
        res_list.append([file, source_wav, destination_wav])
    return res_list


def TestVoiceMaker(source_dir, destination_dir , TestFolder, folder_name=None):
    dst_name = "084_aligned2"
    if not os.path.exists(os.path.join(TestFolder, dst_name, folder_name)):
        os.mkdir(os.path.join(TestFolder, dst_name , folder_name))
    else:
        return
    res_list = load_wavs(source_dir, destination_dir)
    for i in trange(len(res_list)):
        name, src_wav, dst_wav = res_list[i]
        number = name.split(".")[0][-3:]
        #name folderの数字_srcの数字_ファイルの数字.wav
        name = folder_name[-3:] +'_' + "084" + '_' + number + ".wav"
        src = librosa_remixing(src_wav, top_db=50)
        dst = librosa_remixing(dst_wav, top_db=50)
        #it often makes bad effects to b1 filter in dtw
        # window_size = abs(len(src) - len(dst)) * 2
        #zero_cutは共通じゃん…何やってんの（笑）
        # soundfile.write(os.path.join(TestFolder, "SRC_zero_cut", folder_name,  name), src, samplerate=sr)
        # soundfile.write(os.path.join(TestFolder, 'TGT_zero_cut', folder_name,  name), dst, samplerate=sr)
        src_f0, src_t = pyworld.harvest(src, fs=fs)
        dst_f0, dst_t = pyworld.harvest(dst, fs=fs)
        src_sp = pyworld.cheaptrick(src, src_f0, src_t, fs=fs)
        dst_sp = pyworld.cheaptrick(dst, dst_f0, dst_t, fs=fs)
        src_ap = pyworld.d4c(src, src_f0, src_t, fs=fs)
        src_sp_coded = pysptk.sp2mc(src_sp, order=32, alpha=pysptk.util.mcepalpha(fs))
        dst_sp_coded = pysptk.sp2mc(dst_sp, order=32, alpha=pysptk.util.mcepalpha(fs))
        src_sp_coded = trim_zeros_frames(src_sp_coded)
        dst_sp_coded = trim_zeros_frames(dst_sp_coded)
        try:
            res = dtw(src_sp_coded, dst_sp_coded, step_pattern="typeIb")
        except:
            break
        path = res.get_warping_path(target='query')
        src_f0_aligned = src_f0[path]
        src_ap_aligned = src_ap[path, :]
        src_sp_aligned = src_sp[path, :]
        syn_src_wav = pyworld.synthesize(src_f0_aligned, src_sp_aligned, src_ap_aligned, fs=fs)
        # soundfile.write(os.path.join(TestFolder, dst_name, folder_name, name), syn_src_wav, samplerate=sr)
        np.save(os.path.join(TestFolder, dst_name, folder_name, name), syn_src_wav)
        
def VoiceMakers():
    source_base_folder = "C:/Users/Atsuya/Documents/Sites/Voice/jvs_ver1/jvs_ver1"
    folders = os.listdir(source_base_folder)
    Test_folder = "C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices"
    for i in trange(len(folders)):
        print('Folder {} is Starting!'.format(folders[i]))
        source_folder = os.path.join(source_base_folder, folders[i], 'parallel100', 'wav24kHz16bit')
        destination_folder = "C:/Users/Atsuya/PycharmProjects/VoiceConverter/CycleGanConverterExample/TargetVoices/jvs084/parallel100/wav24kHz16bit/"
        TestVoiceMaker(source_folder, destination_folder, Test_folder, folders[i])

def VoiceMakerFile(src_path, dst_path, save_path):
    src_wav = librosa.core.load(src_path, sr=sr)[0]
    dst_wav = librosa.core.load(dst_path, sr=sr)[0]
    src = librosa_remixing(src_wav, top_db=50)
    dst = librosa_remixing(dst_wav, top_db=50)
    src_f0, src_t = pyworld.harvest(src, fs=fs)
    dst_f0, dst_t = pyworld.harvest(dst, fs=fs)
    src_sp = pyworld.cheaptrick(src, src_f0, src_t, fs=fs)
    dst_sp = pyworld.cheaptrick(dst, dst_f0, dst_t, fs=fs)
    src_ap = pyworld.d4c(src, src_f0, src_t, fs=fs)
    src_sp_coded = pysptk.sp2mc(src_sp, order=32, alpha=pysptk.util.mcepalpha(fs))
    dst_sp_coded = pysptk.sp2mc(dst_sp, order=32, alpha=pysptk.util.mcepalpha(fs))
    src_sp_coded = trim_zeros_frames(src_sp_coded)
    dst_sp_coded = trim_zeros_frames(dst_sp_coded)
    try:
        res = dtw(src_sp_coded, dst_sp_coded, step_pattern="typeIb")
        path = res.get_warping_path(target='query')
        src_f0_aligned = src_f0[path]
        src_ap_aligned = src_ap[path, :]
        src_sp_aligned = src_sp[path, :]
        syn_src_wav = pyworld.synthesize(src_f0_aligned, src_sp_aligned, src_ap_aligned, fs=fs)
        soundfile.write(save_path, syn_src_wav, samplerate=sr)
    except:
        print("This file couldn't convert to training data")
        pass

def FileRenaming1():
    source_base_folder = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/ZeroCut_wav24kHz16bit'
    folders = os.listdir(source_base_folder)
    for i in trange(len(folders)):
        folder = folders[i]
        files = os.listdir(os.path.join(source_base_folder, folder))
        for j in trange(len(files)):
            file = files[j]
            src_num = file[3:6]
            # dst_num = file[10:13]
            file_num = file[13:16]
            new_name = src_num + '_' + file_num + '.wav'
            os.rename(os.path.join(source_base_folder, folder, file), os.path.join(source_base_folder, folder, new_name))

def testDataMaker():
    raw_voice_folder = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/my_voices/raw_voices'
    downsampling_folder = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/my_voices/without_alignment'
    alignment_folder = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices/my_voices/alignment'
    files = os.listdir(raw_voice_folder)
    wavs = list()
    for file in files:
        wav = librosa.core.load(os.path.join(raw_voice_folder, file), sr=sr)[0]
        wavs.append((file, wav))
    for file, wav in wavs:
        soundfile.write(os.path.join(downsampling_folder, file), wav, samplerate=sr)
        wav = librosa_remixing(wav, top_db=40)
        soundfile.write(os.path.join(alignment_folder, file), wav, samplerate=sr)

if __name__=='__main__':
    # source_folder = "C:/Users/Atsuya/PycharmProjects/VoiceConverter/CycleGanConverterExample/SourceVoices/jvs034/parallel100/wav24kHz16bit/"
    # destination_folder = "C:/Users/Atsuya/PycharmProjects/VoiceConverter/CycleGanConverterExample/TargetVoices/jvs093/parallel100/wav24kHz16bit/"
    # Test_folder = "C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices"
    # TestVoiceMaker(source_folder, destination_folder, Test_folder)
    VoiceMakers()
    # testDataMaker()
    #This is used for faulted file
    # src_num = 'jvs009'
    # dst_num = 'jvs093'
    # file_num = "100"
    # base_path= "C:/Users/Atsuya/Documents/Sites/Voice/jvs_ver1/jvs_ver1"
    # prefix0 = 'parallel100'
    # prefix1 = 'wav24kHz16bit'
    # file_name = 'VOICEACTRESS100_' + file_num + '.wav'
    # test_folder = "C:/Users/Atsuya/PycharmProjects/VoiceConverter/test-voices"
    # src_path = os.path.join(base_path, src_num, prefix0, prefix1, file_name)
    # dst_path = os.path.join(base_path, dst_num, prefix0, prefix1, file_name)
    # save_path = os.path.join(test_folder, 'SRC_aligned', src_num , src_num + '2' + dst_num + file_num + '.wav')
    # VoiceMakerFile(src_path, dst_path, save_path)
    #File renaming!
    # FileRenaming1()
