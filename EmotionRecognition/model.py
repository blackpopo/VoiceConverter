import tensorflow as tf
import tensorflow_addons as tfa

def residual_block(x, kernel_size, filters, strides):
    initializer =  tf.random_normal_initializer(0., 0.02)
    skip = x
    x = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters, strides=strides, use_bias=False, kernel_initializer=initializer, padding='SAME')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters, strides=strides, use_bias=False, kernel_initializer=initializer, padding='SAME')(x)
    x = x + skip
    return x

def Recognizer(num_classes):
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input([128, 128, 1])
    x = inputs
    x = tf.keras.layers.Conv2D(kernel_size=3, filters=64, strides=2, padding='SAME', kernel_initializer=initializer)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = tf.keras.layers.Conv2D(kernel_size=3, filters=64, strides=2, padding='SAME', kernel_initializer=initializer)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    skip = x
    for i in range(4):
        x = residual_block(x, kernel_size=3, filters=64, strides=1)
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2D(x, kernel_size=3, filters=64, strides=1, padding='SAME', kernel_initializer=initializer, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = x + skip
    x = tf.keras.layers.Conv2D(x, kernel_size=3, filters=64, strides=1, padding='SAME', kernel_initializer=initializer,
                               use_bias=False)(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(num_classes)(x)
    x = tf.keras.layers.Softmax()(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


