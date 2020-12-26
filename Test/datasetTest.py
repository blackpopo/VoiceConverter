import tensorflow as tf
import numpy as np
from tqdm import tqdm


def datasetTest():
    array1 = np.arange(100, 100 + 256*128*100).reshape((-1, 128, 256, 1)).astype(np.float32)
    array2 = np.arange(-100, -100 + 256*128*100).reshape((-1, 128, 256, 1)).astype(np.float32)
    print(array1.shape, array2.shape)
    dataset = tf.data.Dataset.from_tensor_slices((array1, array2))
    dataset = dataset.batch(3)
    for inp, tgt in tqdm(dataset):
        # print(inp, tgt)
        # inp, tgt = dataset[i]
        print(inp.shape, tgt.shape)

def _batch_parser(record_batch):
    # NOTE: Use `tf.parse_example()` to operate on batches of records.
    parsed = tf.io.parse_example(record_batch,
                                 features = {
            "src_f0": tf.io.FixedLenFeature([128], dtype=tf.float32),
            "tgt_f0": tf.io.FixedLenFeature([128], dtype=tf.float32),
            "wav_num": tf.io.FixedLenFeature([], dtype=tf.string)
        })
    src_f0 = tf.expand_dims(parsed["src_f0"], -1)
    tgt_f0 = tf.expand_dims(parsed["tgt_f0"], -1)
    return parsed['src_f0'], parsed['tgt_f0']

# def init_tfrecord_dataset():
    # files_train = glob.glob(DIR_TFRECORDS + '*.tfrecord')
    # random.shuffle(files_train)
    #
    # with tf.name_scope('tfr_iterator'):
    #     ds = tf.data.TFRecordDataset(files_train)  # define data from randomly ordered files
    #     ds = ds.shuffle(buffer_size=10000)  # select elements randomly from the buffer
    #     # NOTE: Change begins here.
    #     ds = ds.batch(BATCH_SIZE,
    #                   drop_remainder=True)  # group elements in batch (remove batch of less than BATCH_SIZE)
    #     ds = ds.map(_batch_parser)  # map batches based on tfrecord format
    #     # NOTE: Change ends here.
    #     ds = ds.repeat()  # iterate infinitely
    #
    #     return ds.make_initializable_iterator()  # initialize the iterator


def f0_single_parse(f0_one):
    features = tf.io.parse_example(
        f0_one,
        features = {
            "src_f0": tf.io.FixedLenFeature([128], dtype=tf.float32),
            "tgt_f0": tf.io.FixedLenFeature([128], dtype=tf.float32),
            "wav_num": tf.io.FixedLenFeature([], dtype=tf.string)
        }
    )
    src_f0 = tf.expand_dims(features["src_f0"], -1)
    tgt_f0 = tf.expand_dims(features["tgt_f0"], -1)
    wav_num = features["wav_num"]
    return wav_num, src_f0, tgt_f0

import glob
def tfrecord_test():
    folder_path = 'C:/Users/Atsuya/PycharmProjects/VoiceConverter/ProductionModel/DataStore/093_084_f0/*'
    files = glob.glob(folder_path)
    #mapをBatchの前に置くと12s,　後に置くと5sでできる！
    dataset = tf.data.TFRecordDataset(files).shuffle(1000).batch(8).map(_batch_parser)
    for sf0, tf0 in tqdm(dataset):
        pass

def tf_rewrite_test():
    pass

def tensor_test():
    tensor = np.array([[[1], [2], [3]]])
    tensor = np.arange(0, 12).reshape((1, 3, 4, 1))
    print(tensor)
    pre = tf.slice(tensor, (0, 0, 0, 0), (tensor.shape[0], tensor.shape[1]-1, tensor.shape[2], tensor.shape[3]))
    post = tf.slice(tensor, (0, 1, 0, 0), (tensor.shape[0], tensor.shape[1]-1, tensor.shape[2], tensor.shape[3]))
    print(post - pre)


if __name__=='__main__':
    # datasetTest()
    # tfrecord_test()
    tensor_test()