from NoUse.dataset import *
import datetime
from tqdm import trange, tqdm
from Spade.spade_loss import *
from Spade.spade_model128to256 import *
from config import SpadeConfig
spade_config = SpadeConfig()
import glob

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

#segmapづくり
#各weight
# @tf.function #discriminatorが動かん！
def train_step(input_image, target, segmap, epoch, encoder, generator, discriminator, encoder_optimizer, generator_optimizer, discriminator_optimizer, summary_writer):
    with tf.GradientTape() as enc_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        input_mean , input_var = encoder(input_image)

        gen_output = generator([target, input_mean, input_var], training=True)

        disc_real_logit = discriminator(input_image, target, training=True)

        disc_fake_logit = discriminator(input_image, gen_output, training=True)

        gen_adv_loss = spade_config.ADV_WEIGHT * generator_loss(disc_fake_logit)

        enc_kl_loss = spade_config.KL_WIGHT * kl_loss(input_mean, input_var)

        gen_feature_loss = spade_config.FEATURE_WEIGHT * feature_loss(disc_real_logit, disc_fake_logit)

        disc_adv_loss = spade_config.ADV_WEIGHT * discriminator_loss(disc_real_logit, disc_fake_logit)

        enc_total_loss = enc_kl_loss + gen_feature_loss

        gen_total_loss = gen_adv_loss + gen_feature_loss

        encoder_gradients = enc_tape.gradient(enc_total_loss, encoder.trainable_variables)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_adv_loss,
                                                     discriminator.trainable_variables)

    encoder_optimizer.apply_gradients(zip(encoder_gradients,
                                          encoder.trainable_variables))
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_adv_loss, step=epoch)
        tf.summary.scalar('enc_kl_loss', enc_kl_loss, step=epoch)
        tf.summary.scalar('gen_feature_loss', gen_feature_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_adv_loss, step=epoch)


# @tf.function #こいつがあるとなんかbatchがおかしくなる！絶対につけちゃダメ！
def fit(dataset, epochs, test_ds1, test_ds2,  save_dir, summary_writer, mode, loading=False):
    if loading:
        encoder = tf.keras.models.load_model(os.path.join(save_dir, 'spade_encoder_' + mode))
        generator = tf.keras.models.load_model(os.path.join(save_dir, 'spade_generator_' + mode))
        discriminator = tf.keras.models.load_model(os.path.join(save_dir, 'spade_discriminator_' + mode))
    elif mode in ['sp', 'sp_full']:
        encoder = Image_encoder()
        generator = Generator()
        discriminator = Discriminator()
    else:
        raise ValueError('Size must be 256 or 513')

    encoder_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=spade_config.BETA1, beta_2=spade_config.BETA2)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=spade_config.BETA1, beta_2=spade_config.BETA2)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=spade_config.BETA1, beta_2=spade_config.BETA2)

    for epoch in trange(epochs):
        # Train
        for input_image, target, segmap in tqdm(dataset.take(1000)):
            train_step(input_image, target, segmap, epoch, encoder, generator, discriminator, encoder_optimizer, generator_optimizer, discriminator_optimizer, summary_writer)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 5 == 0:
            for i, example in enumerate(test_ds1):
                example = np.expand_dims(example, 0)
                # segmap = np_create_segmap(example)
                input_mean, input_var = encoder(example)
                prediction = generator([example, input_mean, input_var])
                l1_loss = tf.reduce_mean(tf.abs(prediction - example))
                with summary_writer.as_default():
                    tf.summary.scalar('test1_gen_l1_loss', l1_loss, step=epoch+i)
                if (i + 1) % 5 == 0:
                    prediction = tf.squeeze(prediction, axis=3)
                    prediction = tf.squeeze(prediction, axis=0)
                    example = np.squeeze(example, axis=3)
                    example = np.squeeze(example, axis=0)
                    visualize(example)
                    visualize(prediction)
                if (i + 1) % 20 == 0:
                    break
            print("Epoch: ", epoch)

            for i, example in enumerate(test_ds2):
                example = np.expand_dims(example, 0)
                # segmap = np_create_segmap(example)
                input_mean, input_var = encoder(example)
                prediction = generator([example, input_mean, input_var])
                l1_loss = tf.reduce_mean(tf.abs(prediction - example))
                with summary_writer.as_default():
                    tf.summary.scalar('test2_gen_l1_loss', l1_loss, step=epoch+i)
                if (i + 1) % 5 == 0:
                    prediction = tf.squeeze(prediction, axis=3)
                    prediction = tf.squeeze(prediction, axis=0)
                    example = np.squeeze(example, axis=3)
                    example = np.squeeze(example, axis=0)
                    visualize(example)
                    visualize(prediction)
                if (i + 1) % 20 == 0:
                    break
            print("Epoch: ", epoch)

            encoder_save_path = os.path.join(save_dir, 'spade_encoder_' + mode)
            encoder.save(encoder_save_path)
            generator_save_path = os.path.join(save_dir, 'spade_generator_' + mode)
            generator.save(generator_save_path)
            discriminator_save_path = os.path.join(save_dir, 'spade_discriminator_' + mode)
            tf.saved_model.save(discriminator, discriminator_save_path)


def np_create_segmap(image):
    mean = np.mean(image)
    segmap = np.zeros_like(image)
    segmap[np.where(image > mean)] = 1
    return segmap
#
# @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
# def tf_create_segmap(input):
#   return tf.numpy_function(np_create_segmap, [input], tf.float32)
#
#
# @tf.function
# def create_segmap(input_image, target_image):
#     segmap = tf_create_segmap(target_image)
#     return input_image, target_image, segmap

def _batch_parser_sp(record_batch):
    # NOTE: Use `tf.parse_example()` to operate on batches of records.
    parsed = tf.io.parse_example(record_batch,
                                 features = {
            "src_sp": tf.io.FixedLenFeature([128, 256], dtype=tf.float32),
            "tgt_sp": tf.io.FixedLenFeature([128, 256], dtype=tf.float32),
            "segmap": tf.io.FixedLenFeature([128, 256], dtype=tf.float32)
        })
    src_f0 = tf.expand_dims(parsed["src_sp"], -1)
    tgt_f0 = tf.expand_dims(parsed["tgt_sp"], -1)
    segmap = tf.expand_dims(parsed["segmap"], -1)
    return src_f0, tgt_f0, segmap

def _batch_parser_sp_full(record_batch):
    # NOTE: Use `tf.parse_example()` to operate on batches of records.
    parsed = tf.io.parse_example(record_batch,
                                 features = {
            "src_sp": tf.io.FixedLenFeature([128, 256], dtype=tf.float32),
            "tgt_sp_full": tf.io.FixedLenFeature([128, 513], dtype=tf.float32),
            "segmap": tf.io.FixedLenFeature([128, 513], dtype=tf.float32)
        })
    src_f0 = tf.expand_dims(parsed["src_sp"], -1)
    tgt_f0 = tf.expand_dims(parsed["tgt_sp"], -1)
    segmap = tf.expand_dims(parsed["segmap"], -1)
    return src_f0, tgt_f0, segmap

def train():
    folder_name = '093_084_alignment'
    file_prefix = '_'.join(['093', '084'])
    test_file_name1 = 'valid_alignment_128_256'
    test_file_name2 = 'valid_without_alignment_128_256'
    mode = 'sp'
    data_path = '../../TestModels/DataStore'
    log_dir = "../log"
    save_dir = '../../TestModels/ModelData'
    summary_writer = tf.summary.create_file_writer(
        log_dir + "/" + mode + '_' + 'spade' + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    ################# Data prepareration ####################
    print('Train DATA Loading... ')

    test_data1 = load_npz(data_path, test_file_name1)
    test_data2 = load_npz(data_path, test_file_name2)

    train_files = glob.glob(os.path.join(data_path, file_prefix + '_' + mode, '*'))
    if mode == 'sp':
        #shuffleがもとのサイズよりでかいとerror
        train_dataset = tf.data.TFRecordDataset(train_files).prefetch(tf.data.experimental.AUTOTUNE).shuffle(10000).batch(spade_config.BATCH_SIZE).map(_batch_parser_sp)
        test_dataset1 = test_data1['sp'].reshape((-1, 128, 256, 1)).astype(np.float32)
        test_dataset2 = test_data2['sp'].reshape((-1, 128, 256, 1)).astype(np.float32)
    elif mode == 'sp_full':
        train_dataset = tf.data.TFRecordDataset(train_files).prefetch(tf.data.experimental.AUTOTUNE).shuffle(
            10000).batch(spade_config.BATCH_SIZE).map(_batch_parser_sp_full)
        test_dataset1 = test_data1['sp_full'].reshape((-1, 128, 513, 1)).astype(np.float32)
        test_dataset2 = test_data2['sp_full'].reshape((-1, 128, 513, 1)).astype(np.float32)
    else:
        raise ValueError('mode must be sp or f0 or sp_full!')

    #train_ds, epochs, test_ds1, test_ds2, checkpoint_dir, checkpoint_prefix, log_dir
    ###################################################
    fit(train_dataset, spade_config.EPOCHS, test_dataset1, test_dataset2, save_dir, summary_writer, mode, loading=False)
# dataset, epochs, test_ds1, test_ds2,  save_dir, summary_writer, mode, loading=False
if __name__=='__main__':
    train()