from config import Pix2PixConfig
config = Pix2PixConfig()
from NoUse.pix2pix_model128to256 import Discriminator128to256
from Spade.spade_model128to256 import Image_encoder, Generator
from Pix2Pix.pix2pix_loss import *
import datetime
from tqdm import trange, tqdm
import glob
from utilities import visualize, load_npz
import os
import numpy as np


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

@tf.function
def train_step(input_image, target, epoch, encoder, generator, discriminator, loss_object, generator_optimizer, discriminator_optimizer, summary_writer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        input_mean , input_var = encoder(input_image)

        gen_output = generator([target, input_mean, input_var], training=True)

        disc_real_output = discriminator([input_image, target], training=True)

        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(loss_object, disc_generated_output, gen_output,
                                                                   target)
        disc_loss = discriminator_loss(loss_object, disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))


    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


# @tf.function #こいつがあるとなんかbatchがおかしくなる！絶対につけちゃダメ！
def fit(dataset, epochs, test_ds1, test_ds2,  save_dir, summary_writer, mode, loading=False):

    if loading:
        encoder = tf.keras.models.load_model(os.path.join(save_dir, 'spade2spade_encoder_' + mode))
        generator = tf.keras.models.load_model(os.path.join(save_dir, 'spade2spade_generator_' + mode))
        discriminator = tf.keras.models.load_model(os.path.join(save_dir, 'spade2spade_discriminator_' + mode))
    elif mode == 'sp':
        encoder = Image_encoder()
        generator = Generator()
        discriminator = Discriminator128to256()
    else:
        raise ValueError('mode must be f0 or sp')

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    for epoch in trange(epochs):
        # Train
        for (input_image, target) in tqdm(dataset):
            train_step(input_image, target, epoch, encoder, generator, discriminator, loss_object, generator_optimizer, discriminator_optimizer, summary_writer)
        print('epoch {} is finished'.format(epoch))


        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 5 == 0:
            for i, example in enumerate(test_ds1):
                example = np.expand_dims(example, 0)
                input_mean, input_var = encoder(example)
                prediction = generator([example, input_mean, input_var], training=False)
                l1_loss = tf.reduce_mean(tf.abs(prediction - example))
                with summary_writer.as_default():
                    tf.summary.scalar('test1_gen_l1_loss', l1_loss, step=epoch+i)
                if (i + 1) % 5 == 0:
                    prediction = tf.squeeze(prediction, axis=3)
                    example = np.squeeze(example, axis=3)
                    prediction = tf.squeeze(prediction, axis=0)
                    example = np.squeeze(example, axis=0)
                    visualize(example)
                    visualize(prediction)
                if (i + 1) % 20 == 0:
                    break
            print("Epoch: ", epoch)

            for i, example in enumerate(test_ds2):
                example = np.expand_dims(example, 0)
                input_mean, input_var = encoder(example)
                prediction = generator([example, input_mean, input_var], training=False)
                l1_loss = tf.reduce_mean(tf.abs(prediction - example))
                with summary_writer.as_default():
                    tf.summary.scalar('test2_gen_l1_loss', l1_loss, step=epoch+i)
                if (i + 1) % 5 == 0:
                    prediction = tf.squeeze(prediction, axis=3)
                    example = np.squeeze(example, axis=3)
                    prediction = tf.squeeze(prediction, axis=0)
                    example = np.squeeze(example, axis=0)
                    visualize(example)
                    visualize(prediction)
                if (i + 1) % 20 == 0:
                    break
            print("Epoch: ", epoch)
            
            encoder_save_path = os.path.join(save_dir, 'spade2spade_encoder_' + mode)
            encoder.save(encoder_save_path)
            generator_save_path = os.path.join(save_dir, 'spade2spade_generator_' + mode)
            generator.save(generator_save_path)
            discriminator_save_path = os.path.join(save_dir, 'spade2spade_discriminator_' + mode)
            discriminator.save(discriminator_save_path)

def _batch_parser_sp_full(record_batch):
    # NOTE: Use `tf.parse_example()` to operate on batches of records.
    parsed = tf.io.parse_example(record_batch,
                                 features = {
            "src_sp": tf.io.FixedLenFeature([128, 513], dtype=tf.float32),
            "tgt_sp_full": tf.io.FixedLenFeature([128, 513], dtype=tf.float32),
        })
    src_f0 = tf.expand_dims(parsed["src_sp"], -1)
    tgt_f0 = tf.expand_dims(parsed["tgt_sp_full"], -1)
    return src_f0, tgt_f0

def _batch_parser_sp(record_batch):
    # NOTE: Use `tf.parse_example()` to operate on batches of records.
    parsed = tf.io.parse_example(record_batch,
                                 features = {
            "src_sp": tf.io.FixedLenFeature([128, 256], dtype=tf.float32),
            "tgt_sp": tf.io.FixedLenFeature([128, 256], dtype=tf.float32),
        })
    src_f0 = tf.expand_dims(parsed["src_sp"], -1)
    tgt_f0 = tf.expand_dims(parsed["tgt_sp"], -1)
    return src_f0, tgt_f0

def _batch_parser_f0(record_batch):
    # NOTE: Use `tf.parse_example()` to operate on batches of records.
    parsed = tf.io.parse_example(record_batch,
                                 features = {
            "src_f0": tf.io.FixedLenFeature([128], dtype=tf.float32),
            "tgt_f0": tf.io.FixedLenFeature([128], dtype=tf.float32)
        })
    src_f0 = tf.expand_dims(parsed["src_f0"], -1)
    tgt_f0 = tf.expand_dims(parsed["tgt_f0"], -1)
    return src_f0, tgt_f0

def train():
    file_prefix = '_'.join(['093', '084'])
    test_file_name1 = 'valid_alignment_128_256'
    test_file_name2 = 'valid_without_alignment_128_256'
    mode = 'f0'
    data_path = '../../TestModels/DataStore'
    log_dir = "../log"
    save_dir = '../../TestModels/ModelData'
    summary_writer = tf.summary.create_file_writer(log_dir  + "/" + mode +'_' + 'spade2spade' + '_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    ################# Data prepareration ####################
    print('Train DATA Loading... ')

    test_data1 = load_npz(data_path, test_file_name1)
    test_data2 = load_npz(data_path, test_file_name2)

    train_files = glob.glob(os.path.join(data_path, file_prefix + '_' + mode, '*'))

    if mode == 'sp':
        train_dataset = tf.data.TFRecordDataset(train_files).prefetch(tf.data.experimental.AUTOTUNE).shuffle(
            1000).batch(config.BATCH_SIZE).map(_batch_parser_sp)
        test_dataset1 = test_data1['sp'].reshape((-1, 128, 256, 1)).astype(np.float32)
        test_dataset2 = test_data2['sp'].reshape((-1, 128, 256, 1)).astype(np.float32)
    elif mode == 'f0':
        train_dataset = tf.data.TFRecordDataset(train_files).prefetch(tf.data.experimental.AUTOTUNE).shuffle(
            1000).batch(config.BATCH_SIZE).map(_batch_parser_f0)
        test_dataset1 = test_data1['f0'].reshape(-1, 128, 1).astype(np.float32)
        test_dataset2 = test_data2['f0'].reshape(-1, 128, 1).astype(np.float32)
    elif mode == 'sp_full':
        train_dataset = tf.data.TFRecordDataset(train_files).prefetch(tf.data.experimental.AUTOTUNE).shuffle(
            1000).batch(config.BATCH_SIZE).map(_batch_parser_sp_full)
        test_dataset1 = test_data1['sp_full'].reshape((-1, 128, 513, 1)).astype(np.float32)
        test_dataset2 = test_data2['sp_full'].reshape((-1, 128, 513, 1)).astype(np.float32)
    else:
        raise ValueError('mode must be sp or f0 or sp_full!')

    #train_ds, epochs, test_ds1, test_ds2, checkpoint_dir, checkpoint_prefix, log_dir
    ###################################################
    fit(train_dataset, config.EPOCHS, test_dataset1, test_dataset2, save_dir, summary_writer, mode)

if __name__=='__main__':
    train()