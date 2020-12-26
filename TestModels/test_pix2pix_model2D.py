import tensorflow as tf
from test_config import Pix2PixConfig
config = Pix2PixConfig()

def FirstdownSampling2D(x, filters, size):
  initializer = tf.random_normal_initializer(0., 0.02)

  x = tf.keras.layers.Conv2D(filters, size, strides=2, padding='valid',
                             kernel_initializer=initializer, use_bias=False)(x)
  x = tf.keras.layers.LeakyReLU()(x)

  return x

#downsample2D のstrideが2kernel_sizeがzize
def downsample2D(x, filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  x = tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False)(x)
  if apply_batchnorm:
    x = tf.keras.layers.BatchNormalization()(x)

  x = tf.keras.layers.LeakyReLU()(x)

  return x

def upsample2D(x, filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  x = tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False)(x)

  x = tf.keras.layers.BatchNormalization()(x)

  if apply_dropout:
      x = tf.keras.layers.Dropout(0.5)(x)

  x = tf.keras.layers.ReLU()(x)

  return x

def dense1D(x, up_down, stride=(2, 2),  apply_dropout=False):
  if apply_dropout:
    x = tf.keras.layers.Dropout(0.5)(x)
  #各frequecyに対する特徴量を学習
  input_shape = x.shape
  x = tf.transpose(x, (0, 2, 1, 3)) #BatchSize, Time, Frequency, channel >> B, F, T, C
  t_stride = stride[0]
  s_stride = stride[1]
  out_channel = 1

  x = tf.reshape(x, (-1, input_shape[2] * input_shape[3])) # F, C

  if up_down=='down':
    x = tf.keras.layers.Dense(input_shape[2] / t_stride / s_stride  * out_channel)(x)
  elif up_down=='up':
    x = tf.keras.layers.Dense(input_shape[2] * t_stride * s_stride  * out_channel)(x)

  x = tf.keras.layers.LeakyReLU()(x)

  if up_down=='down':
    x = tf.reshape(x, [-1, int(input_shape[2] / s_stride), int(input_shape[1] / t_stride), out_channel])
  elif up_down=='up':
    x = tf.reshape(x, [-1, int(input_shape[2] * s_stride), int(input_shape[1] * t_stride), out_channel])


  x = tf.transpose(x, (0, 2, 1, 3))

  return x

def MyGenerator2D():
  inputs = tf.keras.layers.Input(shape=[256, 256, 1])
  # padding_same の場合ceil(input_shape[i] / strides[i])だからセーフ！
  x = inputs
  # first = FirstdownSampling2D(x, 32, 4) #(bs, 256, 256, 32)
  ds1_c = downsample2D(x, 63, 4) # (bs, 128, 128, 63)
  ds1_l = dense1D(x, 'down') #(bs, 128, 128, 1)
  ds1 = tf.keras.layers.concatenate([ds1_c, ds1_l])

  ds2_c = downsample2D(ds1, 127, 4) # (bs, 64, 64, 127)
  ds2_l = dense1D(ds1, 'down') #(bs, 64, 64, 1)
  ds2 = tf.keras.layers.concatenate([ds2_c, ds2_l])

  ds3_c = downsample2D(ds2, 255, 4) # (bs, 32, 32, 256)
  ds3_l = dense1D(ds2, 'down') #(bs, 32, 32, 1)
  ds3 = tf.keras.layers.concatenate([ds3_c, ds3_l])

  ds4_c = downsample2D(ds3, 511, 4) # (bs, 16, 16, 512)
  ds4_l = dense1D(ds3, 'down') #(bs, 16, 16, 1)
  ds4 = tf.keras.layers.concatenate([ds4_c, ds4_l])

  ds5_c = downsample2D(ds4, 511, 4) # (bs, 8, 8, 512)
  ds5_l = dense1D(ds4, 'down') #(bs, 8, 8, 1)
  ds5 = tf.keras.layers.concatenate([ds5_c, ds5_l])

  ds6_c = downsample2D(ds5, 511, 4) # (bs, 4, 4, 512)
  ds6_l = dense1D(ds5, 'down') #(bs, 4, 4, 1)
  ds6 = tf.keras.layers.concatenate([ds6_c, ds6_l])

  ds7_c = downsample2D(ds6, 511, 4) # (bs, 2, 2, 512)
  ds7_l = dense1D(ds6, 'down') #(bs, 2, 2, 1)
  ds7 = tf.keras.layers.concatenate([ds7_c, ds7_l])

  ds8 = downsample2D(ds7, 512, 4) # (bs, 1, 1, 512)

  # us7 = upsample2D(ds8, 512, 4, apply_dropout=True)
  # us7 = tf.keras.layers.concatenate([us7, ds7])
  #
  # us6 = upsample2D(us7, 512, 4, apply_dropout=True)
  # us6 = tf.keras.layers.concatenate([us6, ds6])
  #
  # us5 = upsample2D(us6, 512, 4, apply_dropout=True)
  # us5 = tf.keras.layers.concatenate([us5, ds5])
  #
  # us4 = upsample2D(us5, 512, 4)
  # us4 = tf.keras.layers.concatenate([us4, ds4])
  #
  # us3 = upsample2D(us4, 256, 4)
  # us3 = tf.keras.layers.concatenate([us3, ds3])
  #
  # us2 = upsample2D(us3, 128, 4)
  # us2 = tf.keras.layers.concatenate([us2, ds2])
  #
  # us1 = upsample2D(us2, 64, 4)
  # us1 = tf.keras.layers.concatenate([us1, ds1])

  us7_c = upsample2D(ds8, 511, 4, apply_dropout=True) # (bs, 2, 2, 511)
  us7_l = dense1D(ds8, 'up', apply_dropout=True) #(bs, 2, 2, 1)
  us7 = tf.keras.layers.concatenate([us7_c, us7_l])
  us7 =  tf.keras.layers.concatenate([us7, ds7]) # (bs, 2, 2, 1024)

  us6_c = upsample2D(us7, 511, 4, apply_dropout=True) # (bs, 4, 4, 511)
  us6_l = dense1D(ds7, 'up', apply_dropout=True) #(bs, 4, 4, 1)
  us6 = tf.keras.layers.concatenate([us6_c, us6_l])
  us6 = tf.keras.layers.concatenate([us6, ds6])  # (bs, 4, 4, 1024)

  us5_c = upsample2D(us6, 511, 4, apply_dropout=True) # (bs, 8, 8, 512)
  us5_l = dense1D(us6, 'up', apply_dropout=True) # (bs, 8, 8, 1)
  us5 = tf.keras.layers.concatenate([us5_c, us5_l])
  us5 = tf.keras.layers.concatenate([us5, ds5])  # (bs, 8, 8, 1024)

  us4_c = upsample2D(us5, 511, 4) # (bs, 16, 16, 511)
  us4_l = dense1D(us5, 'up') # (bs, 16, 16, 1)
  us4 = tf.keras.layers.concatenate([us4_c, us4_l])
  us4 = tf.keras.layers.concatenate([us4, ds4])# (bs, 16, 16, 1024)

  us3_c = upsample2D(us4, 255, 4) # (bs, 32, 32, 255)
  us3_l = dense1D(us4, 'up') #(bs, 32, 32, 1)
  us3 = tf.keras.layers.concatenate([us3_c, us3_l])
  us3 = tf.keras.layers.concatenate([us3, ds3])  # (bs, 32, 32, 512)

  us2_c = upsample2D(us3, 127, 4)  #(bs, 64, 64, 127)
  us2_l = dense1D(us3, 'up') #(bs, 64, 64, 1)
  us2 = tf.keras.layers.concatenate([us2_c, us2_l])
  us2 = tf.keras.layers.concatenate([us2, ds2]) # (bs, 64, 64, 256)

  us1_c = upsample2D(us2, 63, 4)  # (bs, 128, 128, 63)
  us1_l = dense1D(us2, 'up') #(bs, 128, 128, 1)
  us1 = tf.keras.layers.concatenate([us1_c, us1_l])
  us1 = tf.keras.layers.concatenate([us1, ds1]) # (bs, 128, 128, 128)

  initializer = tf.random_normal_initializer(0., 0.02)
  x = tf.keras.layers.Conv2DTranspose(config.OUTPUT_CHANNELS, 4 ,strides=2,padding='same',
                                         kernel_initializer=initializer,activation='tanh')(us1) #(bs, 256, 256, 1)

  return tf.keras.Model(inputs=inputs, outputs=x)

def MyDiscriminator2D():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2) #concatenateは最後が追加される

  down1 = downsample2D(x, 64, 4)# (bs, 128, 128, 64)
  down2 = downsample2D(down1, 128, 4) # (bs, 64, 64, 128)
  down3 = downsample2D(down2, 256, 4) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)

  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


if __name__=='__main__':
  generator = MyGenerator2D()
  # tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
  discriminator = MyDiscriminator2D()
  # tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)
