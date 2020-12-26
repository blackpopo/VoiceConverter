import tensorflow as tf
import tensorflow_addons as tfa

def conv2d(inputs, output_channel, kernel_size, strides):
    initializer = tf.random_normal_initializer(0., 0.02)
    return tf.keras.layers.Conv2D(output_channel, kernel_size=kernel_size, strides=strides, kernel_initializer=initializer, use_bias=True)(inputs)

def normalize(inputs, bn=True):
    if bn:
        return tf.keras.layers.BatchNormalization()(inputs)
    else:
        return tfa.layers.InstanceNormalization()(inputs)

def leaky_relu(inputs, alpha):
    return tf.keras.layers.LeakyReLU(alpha=alpha)(inputs)

def relu(inputs):
    return tf.keras.layers.ReLU()(inputs)

def dense_layer(inputs, output_size, activation=None):
    initializer = tf.random_normal_initializer(0., 0.02)
    return tf.keras.layers.Dense(units=output_size, kernel_initializer=initializer, activation=activation)(inputs)

def discriminator_block(inputs, output_channel,kernel_size, strides):
    net = conv2d(inputs, output_channel, kernel_size, strides)(inputs)
    net = normalize(net)
    net = leaky_relu(net, 0.2)
    return net

def residual_block(inputs, output_channel=64, strides=1):
    net = conv2d(inputs, output_channel, 4, strides=strides)
    net = relu(net)
    net = conv2d(net, output_channel, 4, strides=strides)
    net = net + inputs
    return net

def up_2Dsample(x, scale_factor=None):
    if scale_factor is None:
        scale_factor = [2, 2]
    _, h, w, _ = x.shape
    h_scale, w_scale = scale_factor
    return tf.keras.layers.experimental.preprocessing.Resizing(h * h_scale, w * w_scale, interpolation='bilinear')(x)

def bicubic(inputs, width, height):
    return tf.keras.layers.experimental.preprocessing.Resizing(width, height, interpolation='bicubic')(inputs)

def Generator():
    inputs = tf.keras.layers.Input(shape=[128, 128, 1])
    net = conv2d(inputs, 64, 4, 1)
    net = relu(net)
    for i in range(16):
        net = residual_block(net)
    net = up_2Dsample(net, scale_factor=(1, 2))
    net = relu(net)
    net = up_2Dsample(net, scale_factor=(1, 513/256))
    return tf.keras.Model(inputs=inputs, outputs=net)

def Discriminator():
    inputs = tf.keras.layers.Input(shape=[128, 513, 1])
    net = conv2d(inputs, 64, 4, strides=(1, 2))
    net = leaky_relu(net, 0.2)
    net1 = discriminator_block(net, 64, 4, strides=(1, 2)) #128 * 256
    net2 = discriminator_block(net1, 128, 4, strides=(1, 2)) #128 * 128
    net3 = discriminator_block(net2, 256, 4, strides=(2, 2)) #64 * 64
    net4 = discriminator_block(net3, 512, 4, strides=(2, 2)) #32 * 32
    net5 = dense_layer(net4, 1, activation='sigmoid')
    return tf.keras.Model(inputs=[inputs], outputs=[net1, net2, net3, net4, net5])
