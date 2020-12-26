import tensorflow as tf
import tensorflow_addons as tfa
from config import SPConfig

config = SPConfig()

def conv2d_layer(inputs, filters, kernel_size, strides, use_bias=False, activation=None):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=use_bias, padding='same', kernel_initializer=initializer, activation=activation)(inputs)

def norm_layer(inputs):
    if config.BATCH_SIZE == 1:
        x = tfa.layers.InstanceNormalization()(inputs)
    else:
        x = tf.keras.layers.BatchNormalization()(inputs)
    return x

def gated_linear_layer(inputs, gates):
    activation = tf.multiply(inputs, tf.sigmoid(gates))
    return activation

def residual_block2(x, kernel_size, filters, strides):
    skip = x
    x = conv2d_layer(x, kernel_size=kernel_size, filters=filters, strides=strides)
    x = norm_layer(x)
    x = tf.keras.layers.ReLU()(x)
    x = conv2d_layer(x, kernel_size=kernel_size, filters=filters, strides=strides)
    x = norm_layer(x)
    x = tf.keras.layers.ReLU()(x)
    x = x + skip
    return x

def Sp2Sp_Generator(freq_size):
    inputs = tf.keras.layers.Input([128, freq_size, 1])
    x = inputs
    x = conv2d_layer(x, filters=32, kernel_size=3, strides=1, use_bias=True)
    x = tf.keras.layers.ReLU()(x)
    skip = x

    for _ in range(16):
        x = residual_block2(x, kernel_size=3, filters=32, strides=1)

    x = tf.keras.layers.concatenate([x, skip], axis=-1)

    x = conv2d_layer(x, filters=32, kernel_size=3, strides=1)
    x = tf.keras.layers.ReLU()(x)
    x = conv2d_layer(x, filters=config.OUTPUT_CHANNELS, kernel_size=3, strides=1, use_bias=True, activation='sigmoid')
    return tf.keras.models.Model(inputs=inputs, outputs=x)