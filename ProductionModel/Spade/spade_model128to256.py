import tensorflow as tf
import tensorflow_addons as tfa

def conv2D(x, channels, kernel=4, stride=2, pad=0):
    initializer = tf.random_normal_initializer(0., 0.02)
    if pad > 0:
        if type(stride) == tuple:
            h_stride, w_stride = stride
        else:
            h_stride, w_stride = stride, stride
        h = x.shape[1]
        w = x.shape[2]
        if h % h_stride == 0:
            hpad = pad * 2
        else:
            hpad = max(kernel - (h % h_stride), 0)
        if w % w_stride == 0:
            wpad = pad * 2
        else:
            wpad = max(kernel - (w % w_stride), 0)

        pad_top = hpad // 2
        pad_bottom = hpad - pad_top
        pad_left = wpad // 2
        pad_right = wpad - pad_left

        x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    return tf.keras.layers.Conv2D(filters=channels, kernel_size=kernel, strides=stride, kernel_initializer=initializer)(x)

def fully_connected(x, units, use_bias=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Flatten()(x)
    return tf.keras.layers.Dense(units=units, kernel_initializer=initializer, use_bias=use_bias)(x)

def flatten(x):
    return tf.keras.layers.Flatten()(x)

def spade_resblock(segmap, x_init, channels):
    channel_in = x_init.shape[-1]
    channel_middle = min(channel_in, channels)

    x = spade(segmap, x_init, channel_in)
    x = leaky_relu(x, 0.2)
    x = conv2D(x, channels=channel_middle, kernel=3, stride=1, pad=1)
    x = spade(segmap, x, channels=channel_middle)
    x = leaky_relu(x)
    x = conv2D(x, channels=channels, kernel=3, stride=1, pad=1)
    if channel_in != channels:
        x_init = spade(segmap, x_init, channels=channel_in)
        x_init = conv2D(x_init, channels=channels, kernel=1, stride=1)
    return x + x_init


def spade(segmap, x_init, channels):

    x = param_free_norm(x_init)
    _, x_h, x_w, _ = x.shape
    _, segmap_h, segmap_w, _ = segmap.shape

    factor_h = segmap_h // x_h
    factor_w = segmap_w // x_w

    segmap_down = down_sample(segmap, factor_h, factor_w)
    segmap_down = relu(segmap_down)

    segmap_gamma = conv2D(segmap_down, channels=channels, kernel=5, stride=1, pad=2)
    segmap_beta = conv2D(segmap_down, channels=channels, kernel=5, stride=1, pad=2)

    x = x * (1 + segmap_gamma) + segmap_beta
    return x


def param_free_norm(x, epsilon=1e-5):
    x_mean, x_var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    x_std = tf.sqrt(x_var + epsilon)
    return (x - x_mean) / x_std

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.shape
    return tf.keras.layers.experimental.preprocessing.Resizing(h * scale_factor, w * scale_factor, interpolation='bilinear')(x)

def down_sample(x, scale_factor_h, scale_factor_w):
    _, h, w, _ = x.shape
    return tf.keras.layers.experimental.preprocessing.Resizing(h // scale_factor_h, w//scale_factor_w, interpolation="nearest")(x)

def down_sample_average(x, scale_factor=2):
    return tf.keras.layers.AveragePooling2D(strides=scale_factor, pool_size=3, padding='same')(x)

def leaky_relu(x, alpha=0.01):
    return tf.keras.layers.LeakyReLU(alpha=alpha)(x)

def tanh(x):
    return tf.tanh(x)

def instance_norm(x):
    return tfa.layers.InstanceNormalization(epsilon=1e-5, center=True, scale=True)(x)

def relu(x):
    return tf.keras.layers.ReLU()(x)

def z_sample(mean, logvar):
    eps = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)

    return mean + tf.exp(logvar * 0.5) * eps

def Image_encoder(): #cahnnelは64
    channel = 64
    x_init = tf.keras.layers.Input(shape=[128, 256, 1])
    x = conv2D(x_init, channels=channel, stride=(1, 2), pad=1)  # 128 * 128 if 128 * 513 use stride (1, 4)# 128 * 128 if 128 * 513 use stride (1, 4)
    x = instance_norm(x)
    for i in range(3):
        x = leaky_relu(x, 0.2)
        x = conv2D(x, channel * 2, kernel=3, stride=2, pad=1) #64 * 64 >> 32 * 32 >> 16 * 16
        x = instance_norm(x)
        channel = channel * 2

    x = leaky_relu(x, 0.2)
    x = conv2D(x, channel, kernel=3, stride=2, pad=1)
    x = instance_norm(x)
    x = leaky_relu(x, 0.2)
    x = conv2D(x, channel, kernel=3, stride=2, pad=1)
    x = instance_norm(x)
    mean = fully_connected(x, channel//2) #B * 256
    var = fully_connected(x, channel//2) #B * 256
    return tf.keras.Model(inputs=x_init, outputs=[mean, var])

def Generator():
    segmap = tf.keras.layers.Input(shape=[128, 256, 1])
    x_mean = tf.keras.layers.Input(shape=[256])
    x_var = tf.keras.layers.Input(shape=[256])
    channel = 64 * 4 * 4
    batch_size = segmap.shape[0]
    x = z_sample(x_mean, x_var)
    z_width = 128 // pow(2, 6) #2
    z_height = 256 // pow(2, 7) #2

    x = fully_connected(x, units=z_height * z_width * channel)
    x = tf.reshape(x, (-1, z_height, z_width, channel))

    assert x.shape[0] == batch_size, 'x shape{} batch size'.format(x.shape, batch_size)

    x = spade_resblock(segmap, x, channels=channel)

    #up_sampleにconvolutionを使わなくていいの？
    x = up_sample(x, scale_factor=2) #4 * 4
    x = spade_resblock(segmap, x, channels=channel)
    for i in range(5): #8*8 * 1024 >> 16*16 * 512 >> 32*32 *256 >> 64*64 * 128 >> 128 *128 * 64　あと2回文Ok
        x = up_sample(x, scale_factor=2)
        x = spade_resblock(segmap, x, channels=channel//2)
        channel = channel // 2

    #256 version
    x = leaky_relu(x, 0.2)
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2DTranspose(1, 4,
                                             strides=(1, 2),
                                             padding='same',
                                             kernel_initializer=initializer,
                                             activation='tanh')(x)
    return tf.keras.Model(inputs=[segmap, x_mean, x_var], outputs=x)

class Discriminator(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.discriminator_logits = []


    def __call__(self, segmap, x_init, training=None):
        self.discriminator_logits = []
        for scale in range(3):
            feature_loss = []
            channel = 64
            x = tf.concat([segmap, x_init], axis=-1) #128 * 256 * 2
            x = conv2D(x, channel, kernel=4, stride=(1, 2), pad=1) #128 * 128 * 2
            x = leaky_relu(x, 0.2)
            feature_loss.append(x)
            for i in range(1, 4): #64 * 64 > 32 * 32 > 32 * 32
                stride = 1 if i == 4-1 else 2
                x = conv2D(x, channel*2, kernel=4, stride=stride, pad=1)
                x = instance_norm(x)
                x = leaky_relu(x, 0.2)
                channel = min(channel * 2, 512)
            x = conv2D(x, channels=1, kernel=4, stride=1, pad=1)
            feature_loss.append(x)
            self.discriminator_logits.append(feature_loss)
            x_init = down_sample_average(x_init) #64 * 128
            segmap = down_sample_average(segmap) #64  * 128
        return self.discriminator_logits