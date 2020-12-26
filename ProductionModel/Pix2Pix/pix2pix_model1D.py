import tensorflow as tf
from test_config import Pix2PixConfig
config = Pix2PixConfig()
import tensorflow_addons as tfa

#downsample1D のstrideが2kernel_sizeがzize
def downsample1D(filters, size, strides = 2, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv1D(filters, size, strides=strides, padding='same',
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
    tf.keras.layers.Conv1DTranspose(filters, size, strides=2,
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


def Generator1D(activation='sigmoid'):
  inputs = tf.keras.layers.Input(shape=[128, 1])
  #padding_same の場合ceil(input_shape[i] / strides[i])だからセーフ！

  down_stack = [
    downsample1D(64, 4, strides=1,  apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample1D(128, 4), # (bs, 64, 64, 128)
    downsample1D(256, 4), # (bs, 32, 32, 256)
    downsample1D(512, 4), # (bs, 16, 16, 512)
    downsample1D(512, 4), # (bs, 8, 8, 512)
    downsample1D(512, 4), # (bs, 4, 4, 512)
    downsample1D(512, 4), # (bs, 2, 2, 512)
    downsample1D(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample1D(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample1D(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample1D(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample1D(512, 4), # (bs, 16, 16, 1024)
    upsample1D(256, 4), # (bs, 32, 32, 512)
    upsample1D(128, 4), # (bs, 64, 64, 256)
    upsample1D(64, 4), # (bs, 128, 128, 128) #concat済みの大きさ
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv1DTranspose(config.OUTPUT_CHANNELS, 4,strides=1, padding='same',
                                         kernel_initializer=initializer, activation=activation)

  x = inputs
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
  x = last(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator1D():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[128, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[128, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample1D(64, 4, strides=1,  apply_batchnorm=False)(x)
  # down1 = tf.keras.layers.Dropout(0.3)(down1)# (bs, 128, 128, 64)
  down2 = downsample1D(128, 4, strides=1)(down1) # (bs, 64, 64, 128)
  # down2 = tf.keras.layers.Dropout(0.3)(down2)
  down3 = downsample1D(256, 4, strides=1 )(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding1D()(down3) # (bs, 34, 34, 256)
  # conv = (zero_pad1) # (bs, 31, 31, 512)

  conv = tf.keras.layers.Conv1D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)
  if config.BATCH_SIZE == 1:
    batchnorm1 = tfa.layers.InstanceNormalization()(conv)
  else:
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding1D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv1D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)
  return tf.keras.Model(inputs=[inp, tar], outputs=last)

# if __name__=='__main__':
#   generator = Generator()
#   tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)