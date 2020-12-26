import tensorflow as tf
from tensorflow.python.ops import io_ops

# Load Audio File
def load_data(filename:str) -> (list, int):
    wav_loader = io_ops.read_file(filename)

    Audio, SampleRate = tf.audio.decode_wav(wav_loader,
                                   desired_channels=1)

    # channelの次元を削除
    data_ = tf.squeeze(Audio)

    # batch_sizeの次元を追加
    # data__ = tf.expand_dims(data_, axis=0)

    return data_, SampleRate

# compute STFT
def get_stft_spectrogram(data, fft_length=1024):
    # Input: A Tensor of [batch_size, num_samples]
    # mono PCM samples in the range [-1, 1].
    # returns A `[..., frames, fft_unique_bins]`

    stfts = tf.signal.stft(data,
                           frame_length=256,
                           frame_step=512,
                           fft_length=fft_length)

    # 振幅を求める
    spectrograms = tf.abs(stfts)

    return spectrograms

# compute mel-Frequency
def get_mel(stfts):
    n_stft_bin = stfts.shape[-1]          # --> 257 (= FFT size / 2 + 1)

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=128,
        num_spectrogram_bins=n_stft_bin,
        sample_rate=16000,
        lower_edge_hertz=0.0,
        upper_edge_hertz=8000.0
    )
    # --> shape=(257, 128) = (FFT size / 2 + 1, num of mel bins)

    mel_spectrograms = tf.tensordot(
        stfts,                        # (1, 98, 257)
        linear_to_mel_weight_matrix,  # (257, 128)
        1)
    # --> mel_spectrograms shape: (1, 98, 128)

    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    return log_mel_spectrograms

# compute MFCC
# INPUT : (frame_size, mel_bin_size)
# OUTPUT: (frame_size, mfcc_bin_size)
def get_mfcc(log_mel_spectrograms, n_mfcc_bin):

    #n_mfcc_binって取ってくるデータ数か！
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
    mfcc_ = mfcc[..., :n_mfcc_bin]

    return mfcc_

if __name__=="__main__":
    file_path = "../test-voices/my_voices/without_alignment/test2.wav"
    Audio, SampleRate = load_data(file_path)
    print("AudioShape: ", Audio.shape)
    print("SR", SampleRate)
    STFT = get_stft_spectrogram(Audio)
    print("STFT shape", STFT.shape)
    LogMel = get_stft_spectrogram(STFT, 512)
    print("Log mel shape:", LogMel.shape)
    MFCC = get_mfcc(LogMel, -1)
    print("MFCC shape", MFCC.shape)


