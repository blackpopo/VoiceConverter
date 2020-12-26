import tensorflow as tf
from test_config import LSTMConfig
config = LSTMConfig()
from tqdm import trange
import matplotlib.pyplot as plt
import os
import numpy as np
from test_loader import *
import datetime
from itertools import cycle
from DataUtils import Dataset

class Encoder(tf.keras.Model):
  def __init__(self):
    super(Encoder, self).__init__()
    self.batch_sz = config.BATCH_SIZE
    self.enc_units = config.ENC_UNITS
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  # enc_output の shape == (batch_size, max_length, hidden_size)
  def __call__(self, x, hidden):
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(config.DEC_UNITS)
    self.W2 = tf.keras.layers.Dense(config.DEC_UNITS)
    self.V = tf.keras.layers.Dense(1)

  def __call__(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # スコアを計算するためにこのように加算を実行する
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # スコアを self.V に適用するために最後の軸は 1 となる
    # self.V に適用する前のテンソルの shape は  (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights の shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector の合計後の shape == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self):
    super(Decoder, self).__init__()
    self.batch_sz = config.BATCH_SIZE
    self.dec_units = config.DEC_UNITS
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(1)

    # アテンションのため
    self.attention = BahdanauAttention()

  def __call__(self, x, hidden, enc_output):
    # enc_output の shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    #x　のshape == (batch_size, 1)
    context_vector = tf.expand_dims(context_vector, 1)
    # 結合後の x の shape == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([context_vector , x], axis=-1)

    # 結合したベクトルを GRU 層に渡す
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights


def loss_function(loss_object, real, pred):
    return loss_object(real , pred)


def train(train_dataset, test_dataset1, test_dataset2, summary_writer, save_dir, steps_per_epoch):
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.MeanAbsoluteError()
    encoder = Encoder()
    decoder = Decoder()

    for epoch in range(config.EPOCHS):
        print('{} th epoch is starting...'.format(epoch))
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        source_data, target_data = train_dataset.get()

        encoder_save_path = os.path.join(save_dir, 'encoder_model')
        encoder.save_weights(encoder_save_path)
        decoder_save_path = os.path.join(save_dir, 'decoder_model')
        decoder.save_weights(decoder_save_path)

        for i in trange(source_data.shape[0]):
            inp, targ = source_data[i], target_data[i]
            batch = i * steps_per_epoch
            batch_loss = train_step( inp, targ, epoch, encoder, decoder, enc_hidden, loss_object, optimizer, summary_writer)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {}'.format(epoch + 1,
                                                             batch * config.BATCH_SIZE,
                                                             batch_loss))


        for i, inp in enumerate(cycle(test_dataset1)):
            result = evaluate(inp, encoder, decoder)
            plt.plot(result)
            plt.show()
            l1_loss = tf.reduce_mean(tf.abs(inp - result))
            with summary_writer.as_default():
                tf.summary.scalar('test1_gen_l1_loss', l1_loss, step=epoch + i)
            if (i+1)%5==0:
                break

        for i, inp in enumerate(cycle(test_dataset2)):
            result = evaluate(inp, encoder, decoder)
            plt.plot(result)
            plt.show()
            l1_loss = tf.reduce_mean(tf.abs(inp - result))
            with summary_writer.as_default():
                tf.summary.scalar('test1_gen_l1_loss', l1_loss, step=epoch + i)
            if (i+1)%5==0:
                break

        print('Epoch {} Loss {:.4f}'.format(epoch, total_loss))


def evaluate(inputs, encoder, decoder):
    hidden = [tf.zeros((1, config.ENC_UNITS))]
    inputs = tf.expand_dims(inputs, 0)
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([[0.0]], 0)
    result = list()
    for t in range(256):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
        result.append(predictions[0])
    return result


@tf.function
def train_step(inp, targ, epoch, encoder, decoder, enc_hidden, loss_object, optimizer,  summary_writer):
  loss = 0
  with tf.GradientTape() as tape:
    #encoderはinp(BatchSize, TimeLength, 1)を一気に突っ込む
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([0.0] * config.BATCH_SIZE, 1)
    dec_input = tf.expand_dims(dec_input, 1)

    # Teacher Forcing - 正解値を次の入力として供給
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      #decoderは(BatchSize, 1)を突っ込む
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      #loss のshapeは(1)
      loss += loss_function(loss_object, targ[:, t], predictions)
      # Teacher Forcing を使用
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1])) #時間で割る(256)
  print(loss)
  with summary_writer.as_default():
      tf.summary.scalar('train_loss', loss, step=epoch)

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

def main(file_name):
    data_path = './DataStore'
    valid_path = './DataStoreValidation'
    # file_name = 'only_normalized'
    test_file_name1 = 'valid_alignment_log_256_256'
    test_file_name2 = 'valid_without_alignment_log_256_256'
    log_dir = "logs3/"
    save_dir = os.path.join('./ModelDataRNN', file_name)
    summary_writer = tf.summary.create_file_writer(log_dir  + "/" + file_name +'_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    ################# Data prepareration ####################
    print('Train DATA Loading... ')
    train_data = load_npz(data_path, file_name)
    # Dataset そのものにいろいろなデータが入ってんのか…わけわからん。
    # まず、Source or Target>>Batch>>画像データってことか…
    train_source_data = train_data['f0'][:, 0, :].reshape( -1, 256, 1).astype(np.float32)
    train_target_data = train_data['f0'][:, 1, :].reshape(-1, 256, 1).astype(np.float32)
    steps_per_epoch = train_source_data.shape[1] // config.BATCH_SIZE
    train_dataset = Dataset(train_source_data, train_target_data)
    train_dataset.shuffle()
    train_dataset.batch(config.BATCH_SIZE)
    # なんか、preprosessがめちゃくちゃ怪しかった


    # print('Test DATA1 Loading... ')
    test_data1 = load_npz(valid_path, test_file_name1)
    test_dataset1 = test_data1['f0'].reshape(-1, 256,  1).astype(np.float32)
    # test_dataset1 = tf.data.Dataset.from_tensor_slices(test_data1)

    # print('Test DATA2 Loading... ')
    test_data2 = load_npz(valid_path, test_file_name2)
    test_dataset2 = test_data2['f0'].reshape(-1,256, 1).astype(np.float32)
    # test_dataset2 = tf.data.Dataset.from_tensor_slices(test_data2)

    # train_ds, epochs, test_ds1, test_ds2, checkpoint_dir, checkpoint_prefix, log_dir
    ###################################################
    train(train_dataset, test_dataset1, test_dataset2, summary_writer, save_dir, steps_per_epoch)


if __name__=='__main__':
    main('only_normalized_256_256')



