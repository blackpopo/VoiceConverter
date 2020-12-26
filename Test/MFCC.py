import tensorflow as tf
from tensorflow.python.ops import io_ops

def get_mfccs(audio_file=None, signals=None, sample_rate=44100, num_mfccs=13, frame_length=1024, frame_step=512,
              fft_length=1024, fmax=8000, fmin=80):
    """Compute the MFCCs for audio file

    Keyword Arguments:
        audio_file {str} -- audio wav file path (default: {None})
        signals {tensor} -- input signals as tensor or np.array in float32 type (default: {None})
        sample_rate {int} -- sampling rate (default: {44100})
        num_mfccs {int} -- number of mfccs to keep (default: {13})
        frame_length {int} -- frame length to compute STFT (default: {1024})
        つまり1024個数のデータを使って計算、512個飛ばしてSTFT計算？
        frame_step {int} -- frame step to compute STFT (default: {512})
        fft_length {int} -- FFT length to compute STFT (default: {1024})
        fmax {int} -- Top edge of the highest frequency band (default: {8000})
        fmin {int} -- Lower bound on the frequencies to be included in the mel spectrum (default: {80})

    Returns:
        Tensor -- mfccs as tf.Tensor
    """

    if signals is None and audio_file is not None:
        audio_binary = io_ops.read_file(audio_file)
        # tf.contrib.ffmpeg not supported on Windows, refer to issue
        # https://github.com/tensorflow/tensorflow/issues/8271
        waveform, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
        signals = tf.squeeze(waveform)

    # Step 1 : signals->stfts
    # `stfts` is a complex64 Tensor representing the Short-time Fourier Transform of
    # each signal in `signals`. Its shape is [batch_size, ?, fft_unique_bins]
    # where fft_unique_bins = fft_length // 2 + 1 = 513.
    stfts = tf.signal.stft(signals, frame_length=frame_length, frame_step=frame_step,
                                   fft_length=fft_length)
    # Step2 : stfts->magnitude_spectrograms
    # An energy spectrogram is the magnitude of the complex-valued STFT.
    # A float32 Tensor of shape [batch_size, ?, 513].
    magnitude_spectrograms = tf.abs(stfts)

    # Step 3 : magnitude_spectrograms->mel_spectrograms
    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    num_spectrogram_bins = magnitude_spectrograms.shape[-1]

    num_mel_bins = 64

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, fmin,
        fmax)

    mel_spectrograms = tf.tensordot(
        magnitude_spectrograms, linear_to_mel_weight_matrix, 1)

    # Step 4 : mel_spectrograms->log_mel_spectrograms
    log_offset = 1e-6
    log_mel_spectrograms = tf.math.log(mel_spectrograms + log_offset)

    # Step 5 : log_mel_spectrograms->mfccs
    # Keep the first `num_mfccs` MFCCs.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :num_mfccs]
    print(mfccs.shape)

    return mfccs

if __name__=="__main__":
    file_path = "../test-voices/my_voices/without_alignment/test2.wav"
    get_mfccs(file_path)
    #(なんとか, 13)になってんだけど13がわからん！num_mfccs って何者？