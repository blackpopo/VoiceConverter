import tensorflow as tf
from config import SRConfig
import tensorflow_addons as tfa
config = SRConfig()

def residual_block(x, kernel_size, filters, strides):
    initializer =  tf.random_normal_initializer(0., 0.02)
    skip = x
    x = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters, strides=strides, use_bias=False, kernel_initializer=initializer, padding='SAME')(x)
    if config.BATCH_SIZE == 1:
        x = tfa.layers.InstanceNormalization()(x)
    else:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters, strides=strides, use_bias=False, kernel_initializer=initializer, padding='SAME')(x)
    x = x + skip
    return x

def upsampling_block(x, width, kernel_size, filters, strides=1):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters, strides=strides, kernel_initializer=initializer)(x)
    x = tf.keras.layers.experimental.preprocessing.Resizing(x.shape[1], width=width, interpolation='bilinear')(x)
    x = tf.keras.layers.PReLU()(x)
    return x

def upsampling_block2(x, kernel_size, dst_width, dst_channels, strides=1):
    initializer = tf.random_normal_initializer(0., 0.02)
    height = tf.shape[1]
    width = x.shape[2]
    assert dst_width * dst_channels % width == 0
    x = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=dst_width * dst_channels/width, strides=strides, kernel_initializer=initializer, padding='SAME')(x)
    x = tf.reshape(x, (-1, height, dst_width, dst_channels))
    x = tf.keras.layers.PReLU()(x)
    return x

def upsampling_block3(x, kernel_size, filters, strides, padding):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2DTranspose(kernel_size=kernel_size, filters=filters, strides=strides, kernel_initializer=initializer, padding=padding)(x)
    x = tf.keras.layers.PReLU()(x)
    return x

def pixel_2Dshuffler(inputs, shuffle_size=2, name=None):
    n = tf.shape(inputs)[0]
    w1 = tf.shape(inputs)[1]
    w2 = tf.shape(inputs)[2]
    c = inputs.get_shape().as_list()[3]

    oc = c // shuffle_size // shuffle_size
    ow1 = w1 * shuffle_size
    ow2 = w2 * shuffle_size

    outputs = tf.reshape(tensor=inputs, shape=[n, ow1, ow2, oc], name=name)

    return outputs

def Generator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input([128, 128, 1])
    x = inputs
    x = tf.keras.layers.Conv2D(kernel_size=3, filters=64, strides=1, padding='SAME', kernel_initializer=initializer)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    skip = x
    for i in range(16):
        x = residual_block(x, kernel_size=3, filters=64, strides=1)

    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2D(x, kernel_size=3, filters=64, strides=1, padding='SAME', kernel_initializer=initializer, use_bias=False)(x)
    if config.BATCH_SIZE == 1:
        x = tfa.layers.InstanceNormalization()(x)
    else:
        x = tf.keras.layers.BatchNormalization()(x)
    x = x + skip
    # x = upsampling_block(x, 256, kernel_size=3, filters=256)
    # x = upsampling_block(x, 513, kernel_size=3, filters=256)
    # x = upsampling_block2(x, kernel_size=3, dst_width=256, dst_channels=256)
    # x = upsampling_block2(x, kernel_size=3, dst_width=513, dst_channels=256)

    x = upsampling_block3(x, kernel_size=3, filters=256, strides=(1, 2), padding='SAME')(x)
    x = upsampling_block3(x, kernel_size=3, filters=256, strides=(1, 2), padding='VALID')(x)

    x = tf.keras.layers.Conv2D(kernel_size=3, filters=1, strides=1, padding='SAME', activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=x)