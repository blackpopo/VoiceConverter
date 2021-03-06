from config import SPConfig
config = SPConfig()
import datetime
from Sp2Sp import Sp2Sp_Generator
from tqdm import trange, tqdm
from utilities import *
import os
import glob
import random
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

@tf.function
def train_step(input_image, target, epoch, generator, generator_optimizer, summary_writer):
    with tf.GradientTape() as gen_tape:
        gen_output = generator(input_image, training=True)

        gen_loss = tf.reduce_mean(tf.abs(target-gen_output))

        generator_gradients = gen_tape.gradient(gen_loss,
                                                generator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))


    with summary_writer.as_default():
        tf.summary.scalar('gen_l1_loss', gen_loss, step=epoch)


# @tf.function #こいつがあるとなんかbatchがおかしくなる！絶対につけちゃダメ！
def fit(dataset, epochs, test_dataset,  save_dir, summary_writer, mode, loading=False):

    if loading:
        generator = tf.keras.models.load_model(os.path.join(save_dir, 'pix2pix_freq1D_generator_epoch_' + str(90) + '_' + mode))
    elif mode == 'sp':
        generator = Sp2Sp_Generator(513)
    elif mode == 'mel':
        generator = Sp2Sp_Generator(128)
    else:
        raise ValueError('mode must be f0 or sp')

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    for epoch in trange(epochs):
        # Train
        for (input_image, target, _) in tqdm(dataset):
            train_step(input_image, target, epoch, generator, generator_optimizer, summary_writer)
        print('epoch {} is finished'.format(epoch))


        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            for i, data in enumerate(test_dataset):
                src_test, dst_test, _ = data
                prediction = generator(src_test, training=False)
                l1_loss = tf.reduce_mean(tf.abs(prediction - dst_test))
                with summary_writer.as_default():
                    tf.summary.scalar('test1_gen_l1_loss', l1_loss, step=epoch+i)
                if i % 5 == 0:
                    if mode in ['sp', 'mel', 'mel2sp'] : prediction = tf.squeeze(prediction, axis=3);dst_test = np.squeeze(dst_test, axis=3);src_test = tf.squeeze(src_test, axis=3)
                    prediction = tf.squeeze(prediction, axis=0)
                    dst_test = np.squeeze(dst_test, axis=0)
                    src_test = np.squeeze(src_test, axis=0)
                    visualize(src_test, title1='Source curve ' + str(i), title2='Source spectrum '+ str(i))
                    visualize(dst_test, title1='Target curve ' + str(i), title2='Target spectrum '+ str(i))
                    visualize(prediction, title1='Fake curve ' + str(i), title2='Fake spectrum '+ str(i))
                if (i+1) % 20 == 0:
                    break
            print("Epoch: ", epoch)


        generator_save_path = os.path.join(save_dir, 'sp2sp_epoch_' + str(epoch + 1) +'_' + mode)
        generator.save(generator_save_path)

def _batch_parser_sp(record_batch):
    # NOTE: Use `tf.parse_example()` to operate on batches of records.
    parsed = tf.io.parse_example(record_batch,
                                 features = {
            "src_sp": tf.io.FixedLenFeature([128, 256], dtype=tf.float32),
            "dst_sp": tf.io.FixedLenFeature([128, 256], dtype=tf.float32),
            "dst_who": tf.io.FixedLenFeature([], dtype=tf.string)
                                 })
    tgt_who = parsed["dst_who"]
    src_sp = tf.expand_dims(parsed["src_sp"], -1)
    tgt_sp = tf.expand_dims(parsed["dst_sp"], -1)
    return src_sp, tgt_sp, tgt_who

def _batch_parser_sp_full(record_batch):
    # NOTE: Use `tf.parse_example()` to operate on batches of records.
    parsed = tf.io.parse_example(record_batch,
                                 features = {
            "src_sp": tf.io.FixedLenFeature([128, 513], dtype=tf.float32),
            "dst_sp": tf.io.FixedLenFeature([128, 513], dtype=tf.float32),
            "dst_who": tf.io.FixedLenFeature([], dtype=tf.string)
                                 })
    tgt_who = parsed["dst_who"]
    src_sp = tf.expand_dims(parsed["src_sp"], -1)
    tgt_sp = tf.expand_dims(parsed["dst_sp"], -1)
    return src_sp, tgt_sp, tgt_who

def _batch_parser_f0(record_batch):
    # NOTE: Use `tf.parse_example()` to operate on batches of records.
    parsed = tf.io.parse_example(record_batch,
                                 features = {
            "src_f0": tf.io.FixedLenFeature([128], dtype=tf.float32),
            "dst_f0": tf.io.FixedLenFeature([128], dtype=tf.float32),
            "dst_who": tf.io.FixedLenFeature([], dtype=tf.string)
                                 })
    tgt_who = parsed["dst_who"]
    src_f0 = tf.expand_dims(parsed["src_f0"], -1)
    tgt_f0 = tf.expand_dims(parsed["dst_f0"], -1)
    return src_f0, tgt_f0, tgt_who

def _batch_parser_mel(record_batch):
    # NOTE: Use `tf.parse_example()` to operate on batches of records.
    parsed = tf.io.parse_example(record_batch,
                                 features = {
            "src_mel": tf.io.FixedLenFeature([128, 128], dtype=tf.float32),
            "dst_mel": tf.io.FixedLenFeature([128, 128], dtype=tf.float32),
            "dst_who" :tf.io.FixedLenFeature([], dtype=tf.string)
        })
    tgt_who = parsed["dst_who"]
    src_mel = tf.expand_dims(parsed["src_mel"], -1)
    tgt_mel = tf.expand_dims(parsed["dst_mel"], -1)
    return src_mel, tgt_mel, tgt_who

def _batch_parser_mel2sp(record_batch):
    # NOTE: Use `tf.parse_example()` to operate on batches of records.
    parsed = tf.io.parse_example(record_batch,
                                 features = {
            "dst_mel": tf.io.FixedLenFeature([128, 128], dtype=tf.float32),
            "dst_sp": tf.io.FixedLenFeature([128, 513], dtype=tf.float32),
            "dst_who" :tf.io.FixedLenFeature([], dtype=tf.string)
        })
    tgt_who = parsed["dst_who"]
    tgt_mel = tf.expand_dims(parsed["dst_mel"], -1)
    tgt_sp = tf.expand_dims(parsed["dst_sp"], -1)
    return tgt_mel, tgt_sp, tgt_who


def train():
    mode = 'mel'
    #mode がmel or logmel の時はlengthを128に変更。mel2spの時は特殊なモデルを使う
    #logmelとかlog spの時にはmap かければいいや >> と思ったが再構成ができん。
    log_dir = "log"
    save_dir = '../ModelData2'
    summary_writer = tf.summary.create_file_writer(log_dir  + "/" + mode +'_' + 'pix2pix_freq' + '_B' + str(config.BATCH_SIZE) + '_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_files = glob.glob("C:/Users/Atsuya/PycharmProjects/VoiceConverter/ProductionModel/DataStore2/TFRecords2/*_" + mode + '.tfrecords')
    print(' {} files are gotten...'.format(len(train_files)))

    if mode == 'sp':
        train_dataset = tf.data.TFRecordDataset(train_files).prefetch(tf.data.experimental.AUTOTUNE).shuffle(2000).batch(config.BATCH_SIZE).map(_batch_parser_sp_full)
        test_dataset = tf.data.TFRecordDataset(random.sample(train_files, int(len(train_files)*0.1))).batch(1).map(_batch_parser_sp_full)

    elif mode == 'mel':
        train_dataset = tf.data.TFRecordDataset(train_files).prefetch(tf.data.experimental.AUTOTUNE).shuffle(2000).batch(config.BATCH_SIZE).map(_batch_parser_mel)
        test_dataset = tf.data.TFRecordDataset(random.sample(train_files, int(len(train_files)*0.1))).batch(1).map(_batch_parser_mel)

    elif mode == 'mel2sp':
        train_dataset = tf.data.TFRecordDataset(train_files).prefetch(tf.data.experimental.AUTOTUNE).batch(1).shuffle(20000).map(_batch_parser_mel2sp)
        test_dataset = tf.data.TFRecordDataset(random.sample(train_files, int(len(train_files)*0.1))).batch(1).map(_batch_parser_mel2sp)

    else:
        raise ValueError('Mode must be sp or f0')
    # train_ds, epochs, test_ds1, test_ds2, checkpoint_dir, checkpoint_prefix, log_dir
    ###################################################
    fit(train_dataset, config.EPOCHS, test_dataset, save_dir, summary_writer, mode)

if __name__=='__main__':
    train()