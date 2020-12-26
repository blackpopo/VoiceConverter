import tensorflow as tf
from test_config import Pix2PixConfig
config = Pix2PixConfig()
import tensorflow_addons as tfa

def FirstdownSampling2D(filters, size):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='valid',
                             kernel_initializer=initializer, use_bias=False))
  result.add(tf.keras.layers.LeakyReLU())
  return result

#downsample2D のstrideが2kernel_sizeがzize
def downsample2D(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tfa.layers.InstanceNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample2D(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tfa.layers.InstanceNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator2D():
  inputs = tf.keras.layers.Input(shape=[513, 513, 1])
  # padding_same の場合ceil(input_shape[i] / strides[i])だからセーフ！
  first = FirstdownSampling2D(32, 4)
  down_stack = [
    downsample2D(64, 4), # (bs, 128, 128, 64)
    downsample2D(128, 4), # (bs, 64, 64, 128)
    downsample2D(256, 4), # (bs, 32, 32, 256)
    downsample2D(512, 4), # (bs, 16, 16, 512)
    downsample2D(512, 4), # (bs, 8, 8, 512)
    downsample2D(512, 4), # (bs, 4, 4, 512)
    downsample2D(512, 4), # (bs, 2, 2, 512)
    downsample2D(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample2D(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample2D(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample2D(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample2D(512, 4), # (bs, 16, 16, 1024)
    upsample2D(256, 4), # (bs, 32, 32, 512)
    upsample2D(128, 4), # (bs, 64, 64, 256)
    upsample2D(64, 4), # (bs, 128, 128, 128)
  ]
  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(config.OUTPUT_CHANNELS, 3,strides=2,padding='valid',
                                         kernel_initializer=initializer,activation='tanh')

  x = inputs
  x = first(x)
  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)
  skips = reversed(skips[:-1]) #最後のやつ以外足してんのか！
  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])
  x = upsample2D(32, 4)(x)
  x = last(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator2D():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[513, 513, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[513, 513, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down0 = downsample2D(32, 2, False)(x)
  down1 = downsample2D(64, 4)(down0) # (bs, 128, 128, 64)
  down2 = downsample2D(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample2D(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tfa.layers.InstanceNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


# if __name__=='__main__':
  # generator = Generator()
  # tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
  # discriminator = Discriminator()
  # tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)
