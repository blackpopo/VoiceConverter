import tensorflow as tf
import tensorflow_addons as tfa

def gated_linear_layer(inputs, gates):
    activation = tf.multiply(x=inputs, y=tf.sigmoid(gates))
    return activation

def instance_norm_layer(inputs, epsilon=1e-6):
    return tfa.layers.InstanceNormalization(epsilon=epsilon)(inputs)

def batch_norm_layer(inputs):
    return tf.keras.layers.BatchNormalization()(inputs)

def conv1d_layer(inputs, filters, kernel_size, strides, padding='same'):
    initializer = tf.random_normal_initializer(0., 0.02)
    return tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=initializer)(inputs)

def conv2d_layer(inputs, filters, kernel_size, strides, padding='same'):
    initializer = tf.random_normal_initializer(0., 0.02)
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=initializer)(inputs)

def residual1d_block(inputs, filters=512, strides = 1, kernel_size=3):
    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides)
    h1_norm = instance_norm_layer(inputs=h1)
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides)
    h1_norm_gates = instance_norm_layer(inputs=h1_gates)
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates)
    h2 = conv1d_layer(inputs=h1_glu, filters=filters // 2, kernel_size=kernel_size, strides=strides)
    h2_norm = instance_norm_layer(inputs=h2)
    h3 = inputs + h2_norm
    return h3

def upsample2d_block(inputs, filters, kernel_size, strides, shuffle_size=None):
    if shuffle_size is None:
        shuffle_size = [2, 2]
    h1 = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides)
    # h1_shuffle = up_2Dsample(x=h1, scale_factor=shuffle_size) #クソ遅い…。
    h1_shuffle = tf.nn.depth_to_space(input=h1, block_size=2)
    h1_norm = instance_norm_layer(inputs=h1_shuffle)
    h1_gates = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides)
    # h1_shuffle_gates = up_2Dsample(x=h1_gates, scale_factor=shuffle_size)
    h1_shuffle_gates = tf.nn.depth_to_space(input=h1_gates, block_size=2)
    h1_norm_gates = instance_norm_layer(inputs=h1_shuffle_gates)
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates)
    return h1_glu

def up_2Dsample(x, scale_factor=None):
    if scale_factor is None:
        scale_factor = [2, 2]
    _, h, w, _ = x.shape
    h_scale, w_scale = scale_factor
    return tf.keras.layers.experimental.preprocessing.Resizing(h * h_scale, w * w_scale, interpolation='bilinear')(x)

def downsample2d_block(inputs, filters, kernel_size, strides):
    h1 = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides)
    h1_norm = instance_norm_layer(inputs=h1)
    h1_gates = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides)
    h1_norm_gates = instance_norm_layer(inputs=h1_gates)
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates)
    return h1_glu

def CycleGan_generator(freq_len=128):
    inputs = tf.keras.layers.Input([128, freq_len, 1])
    # inputs = tf.transpose(inputs, (0, 2, 1, 3)) #B * C * T * 1
    h1 = conv2d_layer(inputs=inputs, filters=128, kernel_size=[4, 4], strides=[1, 1])
    h1_gates = conv2d_layer(inputs=inputs, filters=128, kernel_size=[4, 4], strides=[1, 1])
    h1_glu = gated_linear_layer(inputs=h1, gates=h1_gates)

    d1 = downsample2d_block(inputs=h1_glu, filters=256, kernel_size=[4, 4], strides=[2, 2]) #32
    d2 = downsample2d_block(inputs=d1, filters=256, kernel_size=[4, 4], strides=[2, 2]) # 8


    #2304は論文のサイズ B * 1 * T/4 * 2304 らしい？　入力　B * 35 * T * 1　が論文。今回は B * 128(T) * 128(C) * 1。
    d3 =  tf.reshape(d2, shape=(-1, d2.shape[1] * d2.shape[2],  d2.shape[3]))
    resh1 = conv1d_layer(inputs=d3, filters=256, kernel_size=1, strides=1)
    resh1_norm = instance_norm_layer(inputs=resh1)

    r1 = residual1d_block(inputs=resh1_norm, filters=512, kernel_size=3)
    r2 = residual1d_block(inputs=r1, filters=512, kernel_size=3)
    r3 = residual1d_block(inputs=r2, filters=512, kernel_size=3)
    r4 = residual1d_block(inputs=r3, filters=512, kernel_size=3)
    r5 = residual1d_block(inputs=r4, filters=512, kernel_size=3)
    r6 = residual1d_block(inputs=r5, filters=512, kernel_size=3)

    resh2 = conv1d_layer(inputs=r6, filters=512, kernel_size=1, strides=1)
    resh2_norm = instance_norm_layer(inputs=resh2)
    resh3 = tf.reshape(resh2_norm, shape=[-1, d2.shape[1], d2.shape[2], d2.shape[3]*2])
    # Upsample
    u1 = upsample2d_block(inputs=resh3, filters=1024, kernel_size=4, strides=[1, 1], shuffle_size=[2, 2])
    u2 = upsample2d_block(inputs=u1, filters=512, kernel_size=4, strides=[1, 1], shuffle_size=[2, 2])

    conv_out = conv2d_layer(inputs=u2, filters=1, kernel_size=[4, 4], strides=[1, 1])
    # conv_out = tf.transpose(conv_out, (0, 2, 1, 3))
    return tf.keras.Model(inputs=inputs, outputs=conv_out) #128 * 128

def CycleGan_discriminator(freq_len=128):
    inputs = tf.keras.layers.Input(shape=[128, freq_len, 1])
    h1 = conv2d_layer(inputs=inputs, filters=128, kernel_size=[4, 4], strides=[1, 1])
    h1_gates = conv2d_layer(inputs=inputs, filters=128, kernel_size=[4, 4], strides=[1, 1])
    h1_glu = gated_linear_layer(inputs=h1, gates=h1_gates)
    d1 = downsample2d_block(inputs=h1_glu, filters=256, kernel_size=[4, 4], strides=[2, 2])
    d2 = downsample2d_block(inputs=d1, filters=512, kernel_size=[4, 4], strides=[2, 2])
    d3 = downsample2d_block(inputs=d2, filters=1024, kernel_size=[4, 4], strides=[2, 2])

    # Output
    o1 = conv2d_layer(inputs=d3, filters=1, kernel_size=[4, 4], strides=[1, 1]) #16 * 16
    return tf.keras.Model(inputs=inputs, outputs=o1)