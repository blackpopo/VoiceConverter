import tensorflow as tf
from test_config import Pix2PixConfig
config = Pix2PixConfig()
import tensorflow_addons as tfa

#downsample1D のstrideが2kernel_sizeがzize
def downsample1D(filters, size, strides=1, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv1D(filters=filters, kernel_size=size, strides=strides, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    if config.BATCH_SIZE == 1:
      result.add(tfa.layers.InstanceNormalization())
    else:
      result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample1D(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv1DTranspose(filters=filters, kernel_size=size, strides=1,
                                    padding='same',
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

def upsampling_block2(x, kernel_size, dst_width, dst_channels, strides=1):
  initializer = tf.random_normal_initializer(0., 0.02)
  height = x.shape[1]
  width = x.shape[2]
  assert dst_width * dst_channels % width == 0
  x = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=dst_width * dst_channels / width, strides=strides,
                             kernel_initializer=initializer, padding='SAME', use_bias=False)(x)
  x = tf.reshape(x, (-1, height, dst_width, dst_channels))
  x = tf.keras.layers.ReLU()(x)
  return x

def upsample1D2(in_length, out_length, out_channel, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  filters = out_channel * out_length / in_length

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv1D(filters=filters, kernel_size=size, strides=1,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(
    tf.keras.layers.Reshape((out_length, out_channel))
  )

  if config.BATCH_SIZE == 1:
    result.add(tfa.layers.InstanceNormalization())
  else:
    result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

#時間方向にのみ畳み込む >> Freqencyの特徴量はkernelサイズで抽出
def GeneratorFreq1D(freq_len=128, upscale=False):

  inputs = tf.keras.layers.Input(shape=[128 , freq_len , 1])

  down_stack = [
    downsample1D(64, 4, apply_batchnorm=False), # (bs, 128, 64)
    downsample1D(128, 4), # (bs, 128,  128)
    downsample1D(256, 4), # (bs, 128,  256)
    downsample1D(512, 4), # (bs, 128, 512)
    downsample1D(512, 4), # (bs, 128, 512)
    downsample1D(512, 4), # (bs, 128, 512)
    downsample1D(512, 4), # (bs, 128, 512)
    downsample1D(512, 4), # (bs, 128, 512)
  ]

  up_stack = [
    upsample1D(512, 4, apply_dropout=True), # (bs, 128, 1024)
    upsample1D(512, 4, apply_dropout=True), # (bs, 128, 1024)
    upsample1D(512, 4, apply_dropout=True), # (bs, 128, 1024)
    upsample1D(512, 4), # (bs, 128,  1024)
    upsample1D(256, 4), # (bs, 128,  512)
    upsample1D(128, 4), # (bs, 128,  256)
    upsample1D(64, 4), # (bs, 128,  256) #concat済みの大きさ
  ]
  
  # up_stack = [
  #   upsample1D2(128, 128, 512, 4, apply_dropout=True),
  #   upsample1D2(128, 128, 512, 4, apply_dropout=True),
  #   upsample1D2(128, 128, 512, 4, apply_dropout=True),
  #   upsample1D2(128, 128, 512, 4),
  #   upsample1D2(128, 128, 256, 4),
  #   upsample1D2(128, 128, 128, 4),
  #   upsample1D2(128, 128, 64, 4)
  # ]

  initializer = tf.random_normal_initializer(0., 0.02)

  pre_first1 = tf.keras.layers.Conv2D(kernel_size=4, filters=32, strides=(1, 1), padding='same', kernel_initializer=initializer, use_bias=False)

  first = tf.keras.layers.Conv2D(filters=config.OUTPUT_CHANNELS, kernel_size=4, kernel_initializer=initializer, strides=1, padding='SAME')

  pre_last = tf.keras.layers.Conv2D(kernel_size=4, filters=32, strides=(1, 1), padding='same', kernel_initializer=initializer, use_bias=False)

  last = tf.keras.layers.Conv2D(filters=config.OUTPUT_CHANNELS, kernel_size=4, kernel_initializer=initializer, strides=1, padding='SAME', activation='sigmoid')


  x = inputs

  x = pre_first1(x)

  x = tf.keras.layers.LeakyReLU()(x)

  first_skip = x

  x = first(x)

  if config.BATCH_SIZE == 1:
    x = tfa.layers.InstanceNormalization()(x)
  else:
    x = tf.keras.layers.BatchNormalization()(x)

  x = tf.keras.layers.LeakyReLU()(x)

  x = tf.squeeze(x, axis=3)

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

  x = tf.expand_dims(x, axis=3)

  x = pre_last(x)

  if config.BATCH_SIZE == 1:
    x = tfa.layers.InstanceNormalization()(x)
  else:
    x = tf.keras.layers.BatchNormalization()(x)

  x = tf.keras.layers.ReLU()(x)

  x = tf.concat([x, first_skip], axis=3)

  x = tf.keras.layers.Conv2D(filters=32, kernel_size=4, padding='SAME', use_bias=True, kernel_initializer=initializer, strides=1)(x)

  x = tf.keras.layers.ReLU()(x)

  x = last(x)

  if upscale:
    x = upsampling_block2(x, kernel_size=3, dst_width=256, dst_channels=256)
    x = upsampling_block2(x, kernel_size=3, dst_width=513, dst_channels=256)

  return tf.keras.Model(inputs=inputs, outputs=x)


#Todo #こいつがなにやってんの？
def DiscriminatorFreq(freq_len=256, time_conv=True):
  initializer = tf.random_normal_initializer(0., 0.02)


  inp = tf.keras.layers.Input(shape=[128, freq_len, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[128, freq_len, 1], name='target_image')
  if time_conv:
    inp = tf.transpose(inp, (0, 2, 1, 3))  # B * T * C * 1 >> B * C * T * 1
    tar = tf.transpose(tar, (0, 2, 1, 3))
  inp = tf.squeeze(inp, axis=3)
  tar = tf.squeeze(tar, axis=3)
  x = tf.keras.layers.concatenate([inp, tar])  # (bs, 128, 256)

  down1 = downsample1D(64, 4, strides=1, apply_batchnorm=False)(x)
  # down1 = tf.keras.layers.Dropout(0.3)(down1)# (bs, 128, 64)
  down2 = downsample1D(128, 4, strides=1)(down1)  # (bs, 128, 128)
  # down2 = tf.keras.layers.Dropout(0.3)(down2)
  down3 = downsample1D(256, 4, strides=1)(down2)  # (bs, 128, 256)

  conv = tf.keras.layers.Conv1D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False, padding='same')(down3)  # (bs, 128, 512)
  if config.BATCH_SIZE == 1:
    batchnorm1 = tfa.layers.InstanceNormalization()(conv)
  else:
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  last = tf.keras.layers.Conv1D(8, 4, strides=1,
                                kernel_initializer=initializer)(leaky_relu)  # (bs, 128, 8)
  last = tf.expand_dims(last, axis=3)
  if time_conv:
    last = tf.transpose(last, perm=(0, 2, 1, 3))
  return tf.keras.Model(inputs=[inp, tar], outputs=last)
