import tensorflow as tf
from test_config import Pix2PixConfig
config = Pix2PixConfig()
import tensorflow_addons as tfa

def downsample(filters, size, strides = 2, apply_batchnorm=True):
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

def upsample(filters, size, strides=2, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
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


def Generator128to256_Sep():
  inputs = tf.keras.layers.Input(shape=[128,256,1])

  down_stack_freq = [
    downsample(64, 4, strides=(1, 2),  apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4, strides=(1, 2)), # (bs, 64, 64, 128)
    downsample(256, 4, strides=(1, 2)), # (bs, 32, 32, 256)
    downsample(512, 4, strides=(1, 2)), # (bs, 16, 16, 512)
    downsample(512, 4, strides=(1, 2)), # (bs, 8, 8, 512)
    downsample(512, 4, strides=(1, 2)), # (bs, 4, 4, 512)
    downsample(512, 4, strides=(1, 2)), # (bs, 2, 2, 512)
    downsample(512, 4, strides=(1, 2)), # (bs, 1, 1, 512)
  ]

  up_stack_freq = [
    upsample(512, 4, strides=(1, 2), apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, strides=(1, 2), apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, strides=(1, 2), apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4, strides=(1, 2)), # (bs, 16, 16, 1024)
    upsample(256, 4, strides=(1, 2)), # (bs, 32, 32, 512)
    upsample(128, 4, strides=(1, 2)), # (bs, 64, 64, 256)
    upsample(64, 4, strides=(1, 2)), # (bs, 128, 128, 128) #concat済みの大きさ
  ]

  down_stack_time = [
    downsample(64, 4, strides=(1, 2),  apply_batchnorm=False), # (bs, 64, 64) 128
    downsample(128, 4, strides=(2, 1)), # (bs, 64,  128) 64
    downsample(256, 4, strides=(2, 1)), # (bs, 32, 32, 256) 32
    downsample(512, 4, strides=(2, 1)), # (bs, 16, 16, 512) 16
    downsample(512, 4, strides=(2, 1)), # (bs, 8, 8, 512) 8
    downsample(512, 4, strides=(2, 1)), # (bs, 4, 4, 512) 4
    downsample(512, 4, strides=(2, 1)), # (bs, 2, 2, 512) 2
    downsample(512, 4, strides=(2, 1)), # (bs, 1, 1, 512) 1
  ]

  up_stack_time = [
    upsample(512, 4, strides=(2, 1), apply_dropout=True), # (bs, 2, 2, 1024) 2
    upsample(512, 4, strides=(2, 1), apply_dropout=True), # (bs, 4, 4, 1024) 4
    upsample(512, 4, strides=(2, 1), apply_dropout=True), # (bs, 8, 8, 1024) 8
    upsample(512, 4, strides=(2, 1)), # (bs, 16, 16, 1024) 16
    upsample(256, 4, strides=(2, 1)), # (bs, 32, 32, 512) 32
    upsample(128, 4, strides=(2, 1)), # (bs, 64, 64, 256) 64
    upsample(64, 4, strides=(2, 1)), # (bs, 128, 128, 128) #concat済みの大きさ 64
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(config.OUTPUT_CHANNELS, 4,
                                         strides=(1, 2),
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x_time = inputs
  x_freq = inputs
  
  #Time convolution
  # Downsampling through the model
  skips_time = []
  for down in down_stack_time:
    x_time = down(x_time)
    skips_time.append(x_time)

  skips_time = reversed(skips_time[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack_time, skips_time):
    x_time = up(x_time)
    x_time = tf.keras.layers.Concatenate()([x_time, skip])
  
  #Frequencty convolution
  # Downsampling through the model
  skips_freq = []
  for down in down_stack_freq:
    x_freq = down(x_freq)
    skips_freq.append(x_freq)

  skips_freq = reversed(skips_freq[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack_freq, skips_freq):
    x_freq = up(x_freq)
    x_freq = tf.keras.layers.Concatenate()([x_freq, skip])


  x = tf.keras.layers.Concatenate()([x_time, x_freq]) # (bs, 128, 128, 256)
  x = downsample(128, 4, strides=(1, 1), apply_batchnorm=False)(x)
  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator128to256_Sep():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[128, 256, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[128, 256, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, (1, 2), False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)

  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  if config.BATCH_SIZE == 1:
    batchnorm1 = tfa.layers.InstanceNormalization()(conv)
  else:
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)
