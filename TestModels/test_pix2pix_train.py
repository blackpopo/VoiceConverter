from test_config import Pix2PixConfig
config = Pix2PixConfig()
from test_pix2pix_model128to256 import *
from test_pix2pix_model513to513 import *
from test_pix2pix_model256to256 import *
from test_pix2pix_model2D import *
from Pix2Pix.pix2pix_model1D import *
from test_utils import *
from test_loader import load_npz
from NoUse.test_pix_augumentation import *
from test_pix2pix_loss import *
import datetime
from tqdm import trange
from itertools import cycle
from DataUtils import Dataset

@tf.function
def train_step(input_image, target, epoch, generator, discriminator, loss_object, generator_optimizer, discriminator_optimizer, summary_writer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

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


# @tf.function
def fit(source_ds, target_ds, epochs, test_ds1, test_ds2,  save_dir, summary_writer, size):
    if size == 513:
        generator = Generator2D()
        discriminator = Discriminator2D()
    elif size == 256:
        generator = Generator256to256()
        discriminator = Discriminator256to256()
        # generator = MyGenerator2D()
        # discriminator = MyDiscriminator2D()
    elif size == 128:
        generator = Generator128to256()
        discriminator = Discriminator128to256()
    else:
        raise ValueError('Size Must be 256 or 513')
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    for epoch in trange(epochs):
        # Train
        for input_image, target in zip(source_ds, target_ds):
            train_step(input_image, target, epoch, generator, discriminator, loss_object, generator_optimizer, discriminator_optimizer, summary_writer)
        print()


        # saving (checkpoint) the model every 20 epochs
        if (epoch+1 ) % 5 == 0:
            for i, example in enumerate(cycle(test_ds1)):
                example = np.expand_dims(example, 0)
                prediction = generator(example, training=False)
                l1_loss = tf.reduce_mean(tf.abs(prediction - example))
                with summary_writer.as_default():
                    tf.summary.scalar('test1_gen_l1_loss', l1_loss, step=epoch+i)
                if (i + 1)%5==0:
                    prediction = tf.squeeze(prediction, axis=3)
                    prediction = tf.squeeze(prediction, axis=0)
                    example = np.squeeze(example, axis=3)
                    example = np.squeeze(example, axis=0)
                    visualize(example)
                    visualize(prediction)
                if (i + 1) % 20 == 0:
                    break
            print("Epoch: ", epoch)

            for i, example in enumerate(cycle(test_ds2)):
                example = np.expand_dims(example, 0)
                prediction = generator(example, training=False)
                l1_loss = tf.reduce_mean(tf.abs(prediction - example))
                with summary_writer.as_default():
                    tf.summary.scalar('test2_gen_l1_loss', l1_loss, step=epoch+i)
                if (i + 1)%5==0:
                    prediction = tf.squeeze(prediction, axis=3)
                    prediction = tf.squeeze(prediction, axis=0)
                    example = np.squeeze(example, axis=3)
                    example = np.squeeze(example, axis=0)
                    visualize(example)
                    visualize(prediction)
                if (i + 1) % 20 == 0:
                    break
            print("Epoch: ", epoch)

        if (epoch + 1) % 50 == 0:
            generator_save_path = os.path.join(save_dir, 'generator_model')
            generator.save_weights(generator_save_path)
            discriminator_save_path = os.path.join(save_dir, 'discriminator_model')
            discriminator.save_weights(discriminator_save_path)



def train(file_name, test_file_name1, test_file_name2, size):
    data_path = './DataStore2'
    valid_path = './DataStoreValidation'
    log_dir = "log3"
    save_dir = os.path.join('./ModelData', file_name)
    summary_writer = tf.summary.create_file_writer(log_dir  + "/" + file_name +'_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    if type(size) == int:
        size_t, size_f = size, size
    elif type(size) == tuple and len(size) == 2:
        size_t, size_f = size
    else:
        raise ValueError('Size must be int or tuple (time length, frequency length!)')
    ################# Data prepareration ####################
    print('Train DATA Loading... ')
    train_data = load_npz(data_path, file_name)
    #Dataset そのものにいろいろなデータが入ってんのか…わけわからん。
    #まず、Source or Target>>Batch>>画像データってことか…
    train_source_data = train_data['sp'][:, 0, :, :].reshape(-1, size_t, size_f, 1).astype(np.float32)
    train_target_data = train_data['sp'][:, 1, :, :].reshape(-1, size_t, size_f, 1).astype(np.float32)
    # train_dataset = tf.data.Dataset.from_tensor_slices([train_source_data, train_target_data])
    # #なんか、preprosessがめちゃくちゃ怪しかった
    # train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # train_dataset = train_dataset.shuffle(config.BUFFER_SIZE)
    # train_dataset = train_dataset.batch(config.BATCH_SIZE)
    train_dataset = Dataset(train_source_data, train_target_data)
    train_dataset.shuffle()
    train_dataset.batch(config.BATCH_SIZE)
    source_ds, target_ds = train_dataset.get()

    # print('Test DATA1 Loading... ')
    test_data1 = load_npz(valid_path, test_file_name1)
    test_dataset1 = test_data1['sp'].reshape(-1, size_t, size_f, 1).astype(np.float32)
    # test_dataset1 = tf.data.Dataset.from_tensor_slices(test_data1)
    # test_dataset1 = test_dataset1.batch(config.BATCH_SIZE)

    # print('Test DATA2 Loading... ')
    test_data2 = load_npz(valid_path, test_file_name2)
    test_dataset2 = test_data2['sp'].reshape(-1, size_t, size_f, 1).astype(np.float32)
    # test_dataset2 = tf.data.Dataset.from_tensor_slices(test_data2)
    # test_dataset2 = test_dataset2.batch(config.BATCH_SIZE)

    #train_ds, epochs, test_ds1, test_ds2, checkpoint_dir, checkpoint_prefix, log_dir
    ###################################################
    fit(source_ds, target_ds, config.EPOCHS, test_dataset1, test_dataset2, save_dir, summary_writer, size_t)

if __name__=='__main__':

    # train('only_normalized_period10_256_256') #_period10_256to256 , _period2, _period2_256to256の3つを試す！
    file_test1_test2_size = [('gaussean_noise', 'valid_alignment', 'valid_without_alignment', 513),
                             ('lowpass', 'valid_alignment', 'valid_without_alignment', 513),
                             ('median_filter', 'valid_alignment', 'valid_without_alignment', 513),
                             ('only_normalized', 'valid_alignment', 'valid_without_alignment', 513),
                             ('only_normalized_256_256', 'valid_alignment_256to256', 'valid_without_alignment_256to256', 256),
                             ('only_normalized_256_256_coded', 'valid_alignment_256to256_coded', 'valid_without_alignment_256to256_coded', 256),
                             ('only_normalized_log_256_256', 'valid_alignment_log_256_256', 'valid_without_alignment_log_256_256', 256),
                             ('only_normalized_period2', 'valid_alignment', 'valid_without_alignment', 513),
                             ('only_normalized_period2_256_256', 'valid_alignment', 'valid_without_alignment', 256),
                             ('only_normalized_period10', 'valid_alignment', 'valid_without_alignment', 513),
                             ('only_normalized10_256_256', 'valid_alignment_256to256', 'valid_without_alignment_256to256', 256),
                             ('only_normalized_without_alignment', 'valid_alignment', 'valid_without_alignment', 513),
                             ('pre_emphasis', 'valid_alignment', 'valid_without_alignment', 513),
                             ('resample_filter', 'valid_alignment', 'valid_without_alignment', 513),
                             ('savgol_filter', 'valid_alignment', 'valid_without_alignment', 513),
                             ('shift', 'valid_alignment', 'valid_without_alignment', 513),
                             ('zero_factor', 'valid_alignment', 'valid_without_alignment', 513),
                             ]

    file_test1_test2_size = [
                            #  ('only_normalized_513_513', 'valid_alignment_513_513', 'valid_without_alignment_513_513', 513),
                            # ('only_normalized_513_513_fp2', 'valid_alignment_513_513_fp2', 'valid_without_alignment_513_513_fp2', 513),
                            # ('only_normalized_513_513_fp10', 'valid_alignment_513_513_fp10', 'valid_without_alignment_513_513_fp10', 513),
                            # ('only_normalized_256_256', 'valid_alignment_256_256', 'valid_without_alignment_256_256', 256),
                            # ('only_normalized_256_256_fp2', 'valid_alignment_256_256_fp2', 'valid_without_alignment_256_256_fp2', 256),
                            ('only_normalized_128_256', 'valid_alignment_128_256', 'valid_without_alignment_128_256', (128, 256)),
                            # ('only_normalized_256_256_fp2', 'valid_alignment_256_256', 'valid_without_alignment_256_256', 256),
                            # ('only_normalized_513_513_log', 'valid_alignment_513_513_log', 'valid_without_alignment_513_513_log', 513),
                            ('only_normalized_256_256_log', 'valid_alignment_256_256_log', 'valid_without_alignment_256_256_log', 256),
                             ]

    # file, valid1, valid2, size = file_test1_test2_size[4]
    # train(file, valid1, valid2, size)

    for file, valid1, valid2, size in file_test1_test2_size:
        print(file)
        train(file, valid1, valid2, size)