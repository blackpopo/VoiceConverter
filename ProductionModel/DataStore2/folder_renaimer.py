import os
from glob import glob

def parallel100(source_folder):
    parallel100_folders = os.listdir(source_folder)
    for folder in parallel100_folders:
        number = folder[:3]
        files = os.listdir(os.path.join(source_folder, folder))
        for file in files:
            wav_num = file.split('.')[0].strip('_')[-3:]
            new_name = number + '_' + wav_num + '.wav'
            old_path = os.path.join(source_folder, folder, file)
            new_path = os.path.join(source_folder, folder, new_name)
            os.rename(old_path, new_path)

def nonpara30(source_folder):
    numbers = os.listdir(source_folder)
    for number in numbers:

        one_folder = os.path.join(source_folder, number, number + '_nonpara30')
        the_other_folder = os.path.join(source_folder, number, '127_nonpara30')
        one_files = os.listdir(one_folder)
        the_other_files = os.listdir(the_other_folder)

        for file in one_files:
            wav_num =  file.split('.')[0].strip('_').replace('-', '_').split('_')[-1]
            new_name = number +'_' + wav_num +'.wav'
            old_path = os.path.join(one_folder, file)
            new_path = os.path.join(one_folder, new_name)
            os.rename(old_path, new_path)

        for file in the_other_files:
            wav_num =  file.split('.')[0].strip('_').replace('-', '_').split('_')[-1]
            new_name = '127_' + wav_num +'.wav'
            old_path = os.path.join(the_other_folder, file)
            new_path = os.path.join(the_other_folder, new_name)
            os.rename(old_path, new_path)

if __name__=='__main__':
    parallel100_source_folder = 'C:/Users/Atsuya/Documents/SoundRecording/parallel100'
    nonpara30_source_folder = 'C:/Users/Atsuya/Documents/SoundRecording/nonpara30'
    parallel100(parallel100_source_folder)
    nonpara30(nonpara30_source_folder)