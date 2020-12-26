import tensorflow as tf
from config import SRConfig
import tensorflow_addons as tfa
config = SRConfig()

def residual_block1D(x, kernel_size, filters, strides):
    initializer =  tf.random_normal_initializer(0., 0.02)
    skip = x
    x = tf.keras.layers.Conv1D(kernel_size=kernel_size, filters=filters, strides=strides, use_bias=False, kernel_initializer=initializer, padding='SAME')(x)
    if config.BATCH_SIZE == 1:
        x = tfa.layers.InstanceNormalization()(x)
    else:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = tf.keras.layers.Conv1D(kernel_size=kernel_size, filters=filters, strides=strides, use_bias=False, kernel_initializer=initializer, padding='SAME')(x)
    x = x + skip
    return x

def upsampling_block1D(x, width, kernel_size, filters, strides=1):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv1D(kernel_size=kernel_size, filters=filters, strides=strides, kernel_initializer=initializer, padding='SAME')(x)
    x = tf.keras.layers.experimental.preprocessing.Resizing(x.shape[1], width=width, interpolation='bilinear')(x)
    x = tf.keras.layers.PReLU()(x)
    return x

def upsampling_block21D(x, kernel_size, dst_length, dst_channels, strides=1):
    initializer = tf.random_normal_initializer(0., 0.02)
    length = x.shape[1]
    assert dst_length * dst_channels % length == 0
    x = tf.keras.layers.Conv1D(kernel_size=kernel_size, filters=dst_length * dst_channels / length, strides=strides, kernel_initializer=initializer, padding='SAME', use_bias=False)(x)
    x = tf.reshape(x, (-1, dst_length, dst_channels))
    x = tf.keras.layers.PReLU()(x)
    return x

def upsampling_block(x, width, kernel_size, filters, strides=1):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters, strides=strides, kernel_initializer=initializer, padding='SAME', use_bias=False)(x)
    x = tf.keras.layers.experimental.preprocessing.Resizing(x.shape[1], width=width, interpolation='bilinear')(x)
    x = tf.keras.layers.PReLU()(x)
    return x

def upsampling_block2(x, kernel_size, dst_width, dst_channels, strides=1):
    initializer = tf.random_normal_initializer(0., 0.02)
    height = x.shape[1]
    width = x.shape[2]
    assert dst_width * dst_channels % width == 0
    x = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=dst_width * dst_channels /width, strides=strides, kernel_initializer=initializer, padding='SAME', use_bias=False)(x)
    x = tf.reshape(x, (-1, height, dst_width, dst_channels))
    x = tf.keras.layers.PReLU()(x)
    return x

def upsampling_block3(x, kernel_size, filters, strides, padding):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2DTranspose(kernel_size=kernel_size, filters=filters, strides=strides, kernel_initializer=initializer, padding=padding)(x)
    x = tf.keras.layers.PReLU()(x)
    return x

def Generator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input([128, 128, 1])
    x = inputs

    x = tf.keras.layers.Conv2D(kernel_size=3, filters=64, strides=1, padding='SAME', kernel_initializer=initializer)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)

    skip = x

    x = tf.keras.layers.Conv2D(kernel_size=3, filters=1, strides=1, padding='SAME', kernel_initializer=initializer)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)

    x = tf.squeeze(x, axis=3)

    for i in range(16):
        x = residual_block1D(x, kernel_size=3, filters=128, strides=1)

    x = tf.keras.layers.Conv1D(kernel_size=3, filters=128, strides=1, padding='SAME', kernel_initializer=initializer, use_bias=False)(x)

    if config.BATCH_SIZE == 1:
        x = tfa.layers.InstanceNormalization()(x)
    else:
        x = tf.keras.layers.BatchNormalization()(x)

    #UpSampling in 1D or 2D?

    x = tf.expand_dims(x, axis=3)

    x = tf.keras.layers.Conv2D(kernel_size=3, filters=64, strides=1, padding='SAME', kernel_initializer=initializer)(x)
    x = tf.keras.layers.PReLU()(x)

    x = tf.concat([x, skip], axis=-1)

    x = tf.keras.layers.Conv2D(kernel_size=3,filters=32, strides=1, padding='SAME', kernel_initializer=initializer)(x)
    x = tf.keras.layers.PReLU()(x)

    x = upsampling_block2(x, kernel_size=3, dst_width=256, dst_channels=256)
    x = upsampling_block2(x, kernel_size=3, dst_width=513, dst_channels=256)

    x = tf.keras.layers.Conv2D(kernel_size=3, filters=1, strides=1, padding='SAME', kernel_initializer=initializer,  activation='sigmoid')(x)


    return tf.keras.Model(inputs=inputs, outputs=x)