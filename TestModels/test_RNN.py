import tensorflow as tf
from test_config import LSTMConfig
config = LSTMConfig()
from tqdm import trange

#Data の形は？　#BAtchSize * 文字列の長さ？
#shape が (batch_size, max_length, hidden_size) のエンコーダー出力

def Encoder():
    inputs = tf.keras.layers.Input(shape=[config.TIMESTEP, 128])
    gru = tf.keras.layers.GRU(config.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    outputs = gru(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def initialize_hidden_state():
    return tf.zeros((config.BATCH_SIZE, config.ENC_UNITS))


def BahdanauAttention():
    W1 = tf.keras.layers.Dense(config.DEC_UNITS)
    W2 = tf.keras.layers.Dense(config.DEC_UNITS)
    V = tf.keras.layers.Dense(1)
    query = tf.keras.layers.Input(shape=[])
    values = tf.keras.layers.Input(shape=[])
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # スコアを計算するためにこのように加算を実行する
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # スコアを self.V に適用するために最後の軸は 1 となる
    # self.V に適用する前のテンソルの shape は  (batch_size, max_length, units)
    score = V(tf.nn.tanh(
        W1(values) + W2(hidden_with_time_axis)))

    # attention_weights の shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector の合計後の shape == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return tf.keras.Model(inputs=[query, values], outputs=[context_vector, attention_weights])

def Decoder():
    hidden = tf.keras.layers.Input(shape=[])
    enc_output = tf.keras.layers.Input(shape=[])
    x = tf.keras.layers.Input(shape=[])
    gru = tf.keras.layers.GRU(config.DEC_UNITS,
                             return_sequences=True,
                             return_state=True,
                             recurrent_initializer='glorot_uniform')
    conv1 = tf.keras.layers.Conv1D(filters=8, kernel_size=10,
              strides=1, padding="same", activation="relu",)
    conv2 = tf.keras.layers.Conv1D(filters=1, kernel_size=8, padding='same', activation='tanh' )
    max_pooling = tf.keras.layers.GlobalMaxPooling1D()
    # アテンションのため
    attention = BahdanauAttention()
    # enc_output の shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = attention(hidden, enc_output)

    # 結合後の x の shape == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # 結合したベクトルを GRU 層に渡す
    output, state = gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2], 1))

    # output shape == (batch_size, vocab)
    x = conv1(output)
    x = conv2(x)
    max_pooling(x)
    return x, state, attention_weights

def l1_loss(y, y_pred):
  l1_loss = tf.reduce_mean(tf.abs(y-y_pred))
  return l1_loss

def train(dataset):
    optimizer = tf.keras.optimizers.Adam()
    EPOCHS = 10
    encoder = Encoder()
    decoder = Decoder()


    for epoch in range(EPOCHS):

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for batch in trange(dataset):
            inp, targ = dataset[batch]
            #train_step(inp, targ, encoder, decoder, loss_function, optimizer, enc_hidden)
            batch_loss = train_step(inp, targ, encoder , decoder,  optimizer,  enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))



@tf.function
def train_step(inp, targ, encoder, decoder,  optimizer, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([0.0] * config.BATCH_SIZE, 1)

    # Teacher Forcing - 正解値を次の入力として供給
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += l1_loss(targ[:, t], predictions)

      # Teacher Forcing を使用
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss