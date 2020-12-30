import tensorflow as tf
import os
from tqdm import trange, tqdm
import glob
from model import Recognizer
import random
from datetime import datetime

@tf.function
def train_step(input_image, target, epoch, recognizer , optimizer, summary_writer):
    with tf.GradientTape() as tape:
        output = recognizer(input_image, training=True)

        losses = loss(target, output)

        gradients = tape.gradient(losses, recognizer.trainable_variables)

    optimizer.apply_gradients(zip(gradients, recognizer.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=epoch)

def loss(labels, logits):
    cross_entropy = tf.losses.categorical_crossentropy(labels, logits)
    return tf.reduce_mean(cross_entropy)

def fit(dataset, epochs, test_dataset,  save_dir, summary_writer, model_name, loading=False, start_epoch=0):

    if loading:
        recognizer = tf.keras.models.load_model(os.path.join(save_dir, model_name +'_' + start_epoch))
    else:
        recognizer = Recognizer(10)

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    for epoch in trange(start_epoch, epochs):
        # Train
        for (input_image, target, _) in tqdm(dataset):
            train_step(input_image, target, epoch, recognizer, generator_optimizer, summary_writer)
        print('epoch {} is finished'.format(epoch))


        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            for i, data in enumerate(test_dataset):
                labels, sounds = data
                logits = recognizer(sounds, training=False)
                losses = loss(labels, logits)
                with summary_writer.as_default():
                    tf.summary.scalar('test_loss', losses, step=epoch+i)
                if i % 5 == 0:
                    pass
            print("Epoch: ", epoch)


            save_path = os.path.join(save_dir, model_name +'_' + start_epoch)
            recognizer.save(save_path)

def train(loading=False, start_epoch=0):
    #mode がmel or logmel の時はlengthを128に変更。mel2spの時は特殊なモデルを使う
    #logmelとかlog spの時にはmap かければいいや >> と思ったが再構成ができん。正規化すると
    log_dir = "log"
    save_dir = '../ModelData2'
    model_name = 'feeling'
    summary_writer = tf.summary.create_file_writer(log_dir  + "/"  + model_name  + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    epochs = 200
    train_files = glob.glob("C:/Users/Atsuya/PycharmProjects/VoiceConverter/ProductionModel/DataStore2/TFRecords_{}/*_".format(mode) + mode + '.tfrecords')
    print(' {} files are gotten...'.format(len(train_files)))

    train_dataset = tf.data.TFRecordDataset(train_files).prefetch(tf.data.experimental.AUTOTUNE).shuffle(len(train_files)).batch(config.BATCH_SIZE).map(_batch_parser_sp).map(_batch_parser_extract).map(_batch_parser_normalize95)
    test_dataset = tf.data.TFRecordDataset(random.sample(train_files, int(len(train_files)*0.1))).batch(1).map(_batch_parser_sp).map(_batch_parser_extract).map(_batch_parser_normalize95)
    # train_ds, epochs, test_ds1, test_ds2, checkpoint_dir, checkpoint_prefix, log_dir
    ###################################################
    fit(train_dataset, epochs, test_dataset,  save_dir, summary_writer, model_name, loading=loading, start_epoch=start_epoch)



if __name__=='__main__':
    train()
