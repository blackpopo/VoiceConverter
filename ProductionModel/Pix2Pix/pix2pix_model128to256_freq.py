import tensorflow as tf
from test_config import Pix2PixConfig
config = Pix2PixConfig()
import tensorflow_addons as tfa

def downsample(filters, size, strides, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    if config.BATCH_SIZE == 1:
      result.add(tfa.layers.InstanceNormalization())
    else:
      result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, strides, padding="same", apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                    padding=padding,
                                    kernel_initializer=initializer,
                                    use_bias=False))


  if config.BATCH_SIZE == 1:
    result.add(tfa.layers.InstanceNormalization())
  else:
    result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def upsample2(kernel_size, in_height, out_height, in_width, out_width, output_channel, padding="same", apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  filters = output_channel * out_width * out_height / in_height/ in_width

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=1,
                                    padding=padding,
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(
    tf.keras.layers.Reshape((out_height, out_width, output_channel))
  )


  if config.BATCH_SIZE == 1:
    result.add(tfa.layers.InstanceNormalization())
  else:
    result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

# #時間方向にのみ畳み込む >> Freqencyの特徴量はkernelサイズで抽出
def GeneratorFreq(freq_len=256, activation="sigmoid", use_upscale=False):
  inputs = tf.keras.layers.Input(shape=[128, freq_len ,1])

  if freq_len / 128 > 1:
    down_stack = [
      downsample(32, (3, 3), strides=(1, 1),  apply_batchnorm=False), # (bs, 128, 128, 64)
      downsample(32, (3, 3), strides=(1, 1)),  # (bs, 64, 64, 128)
      downsample(32, (3, 3), strides=(1, 1)),  # (bs, 32, 32, 256)
      downsample(32, (3, 3), strides=(1, 1)),  # (bs, 16, 16, 512)
      downsample(32, (3, 3), strides=(1, 1)),  # (bs, 8, 8, 512)
      downsample(32, (3, 3), strides=(1, 1)),  # (bs, 4, 4, 512)
      downsample(32, (3, 3), strides=(1, 1)),  # (bs, 2, 2, 512)
      # downsample(32, (3, 3), strides=(1, 1)), # (bs, 1, 1, 512)
    ]

    up_stack = [
      # upsample(32, (3, 3), strides=(1, 1), apply_dropout=True), # (bs, 2, 2, 1024)
      upsample(32, (3, 3), strides=(1, 1), apply_dropout=True),  # (bs, 4, 4, 1024)
      upsample(32, (3, 3), strides=(1, 1), apply_dropout=True),  # (bs, 8, 8, 1024)
      upsample(32, (3, 3), strides=(1, 1)),  # (bs, 16, 16, 1024)
      upsample(32, (3, 3), strides=(1, 1)),  # (bs, 32, 32, 512)
      upsample(32, (3, 3), strides=(1, 1)),  # (bs, 64, 64, 256)
      upsample(32, (3, 3), strides=(1, 1)), # (bs, 128, 128, 128) #concat済みの大きさ
    ]

  else:
    down_stack = [
      downsample(32, (3, 3), strides=(1, 1),  apply_batchnorm=False), # (bs, 128, 128, 64)
      downsample(32, (3, 3), strides=(1, 1)), # (bs, 64, 64, 128)
      downsample(32, (3, 3), strides=(1, 1)), # (bs, 32, 32, 256)
      downsample(32, (3, 3), strides=(1, 1)), # (bs, 16, 16, 512)
      downsample(32, (3, 3), strides=(1, 1)), # (bs, 8, 8, 512)
      downsample(32, (3, 3), strides=(1, 1)), # (bs, 4, 4, 512)
      downsample(32, (3, 3), strides=(1, 1)), # (bs, 2, 2, 512)
      downsample(32, (3, 3), strides=(1, 1)), # (bs, 1, 1, 512)
    ]

    up_stack = [
      upsample(32, (3, 3), strides=(1, 1), apply_dropout=True), # (bs, 2, 2, 1024)
      upsample(32, (3, 3), strides=(1, 1), apply_dropout=True), # (bs, 4, 4, 1024)
      upsample(32, (3, 3), strides=(1, 1), apply_dropout=True), # (bs, 8, 8, 1024)
      upsample(32, (3, 3), strides=(1, 1)), # (bs, 16, 16, 1024)
      upsample(32, (3, 3), strides=(1, 1)), # (bs, 32, 32, 512)
      upsample(32, (3, 3), strides=(1, 1)), # (bs, 64, 64, 256)
      upsample(32, (3, 3), strides=(1, 1)), # (bs, 128, 128, 128) #concat済みの大きさ
    ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(config.OUTPUT_CHANNELS, (4, 4),
                                         strides=(1, 1),
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation=activation) # (bs, 256, 256, 3)

  x = inputs

  if freq_len/128 > 1:
    x = downsample(32, (3, 3), strides=(1, 1), apply_batchnorm=False)(x)
    first_skip = x
    x = downsample(32, (3, 3), strides=(1, int(freq_len/128)), apply_batchnorm=False)(x)
    second_skip = x

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  if use_upscale:
    x = upsample2(kernel_size=4, in_height=128, in_width=128, out_height=128, out_width=256, output_channel=64)(x)
    x = upsample2(kernel_size=4, in_height=128, in_width=256, out_height=128, out_width=513, output_channel=32)(x) #128 >> 256 >> 513

  if freq_len/128 > 1:
    x = upsample(32, (3, 3), strides=(1, 1), apply_dropout=False)(x)
    x = tf.keras.layers.Concatenate()([x, second_skip])
    x = upsample(32, (3, 3), strides=(1, int(freq_len/128)), apply_dropout=False)(x)
    x = tf.keras.layers.Concatenate()([x, first_skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


#Todo #こいつがなにやってんの？
def DiscriminatorFreq(freq_len=256):
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[128, freq_len, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[128, freq_len, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 128, 256, channels*2)

  down1 = downsample(64, (4, 2), (2, 1), False)(x) # (bs, 64, 256, 64)
  down1 = tf.keras.layers.Dropout(0.5)(down1)
  down2 = downsample(128, (4, 2), (2, 1))(down1) # (bs, 32, 256, 128)
  down2 = tf.keras.layers.Dropout(0.25)(down2)
  down3 = downsample(256, (4, 2), (2, 1))(down2) # (bs, 16, 256, 256)
  down3 = tf.keras.layers.Dropout(0.125)(down3)
  down4 = downsample(256, (4, 2), (2, 1))(down3) # (bs, 8, 256, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D((1, 0))(down4) # (bs, 10, 256, 256)

  conv = tf.keras.layers.Conv2D(512, (4, 1), strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 7, 256, 256)

  if config.BATCH_SIZE == 1:
    batchnorm1 = tfa.layers.InstanceNormalization()(conv)
  else:
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D((1, 0))(leaky_relu) # (bs, 9, 256, 256)

  #30 * 30 の画像分割してそれぞれが正しいか当てている？ >> 6(0.1s) * 256 の分割をして判定させよう！
  last = tf.keras.layers.Conv2D(1, (4, 1), strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 6, 256, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)
