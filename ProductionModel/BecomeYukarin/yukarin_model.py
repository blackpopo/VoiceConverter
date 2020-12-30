import tensorflow as tf

def CBR(out_ch, bn=True, sample='down', activation='relu', dropout=False):
    result = tf.keras.Sequential()

    initializer = tf.random_normal_initializer(0.0, 0.02)
    if sample == 'down':
       result.add(tf.keras.layers.Conv1D(filters=out_ch, kernel_size=3, strides=2, use_bias=True, padding='same', kernel_initializer=initializer))
    elif sample == 'up':
        result.add(tf.keras.layers.Conv1DTranspose(filters=out_ch, kernel_size=3, strides=2, use_bias=True, padding='same', kernel_initializer=initializer))
    else:
        result.add(tf.keras.layers.Conv1D(filters=out_ch, kernel_size=1, strides=2, use_bias=True, padding='same', kernel_initializer=initializer))
    if bn:
        result.add(tf.keras.layers.BatchNormalization())
    if dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    if activation == 'relu':
        result.add(tf.keras.layers.ReLU())
    elif activation == 'leaky_relu':
        result.add(tf.keras.layers.LeakyReLU())
    return result


def Generator(base=64, extensive_layers=8):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    inputs = tf.keras.layers.Input([128, 128])

    x = inputs

    if extensive_layers > 0:
        x = tf.keras.layers.Conv1D(base * 1, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)(x)
    else:
        x = tf.keras.layers.Conv1D(base * 1, kernel_size=1, strides=1, padding='same', kernel_initializer=initializer)(x)

    x = tf.keras.layers.LeakyReLU()(x)
    skips = [x]

    down_stack = [
        CBR(base*2, bn=True, sample='down', activation='leaky_relu', dropout=False),
        CBR(base*4, bn=True, sample='down', activation='leaky_relu', dropout=False),
        CBR(base*8, bn=True, sample='down', activation='leaky_relu', dropout=False),
        CBR(base*8, bn=True, sample='down', activation='leaky_relu', dropout=False),
        CBR(base*8, bn=True, sample='down', activation='leaky_relu', dropout=False),
        CBR(base*8, bn=True, sample='down', activation='leaky_relu', dropout=False),
        CBR(base*8, bn=True, sample='down', activation='leaky_relu', dropout=False),
    ]

    up_stack = [
        CBR(base*8, bn=True, sample='up', activation='relu', dropout=True),
        CBR(base*8, bn=True, sample='up', activation='relu', dropout=True),
        CBR(base*8, bn=True, sample='up', activation='relu', dropout=True),
        CBR(base*8, bn=True, sample='up', activation='relu', dropout=False),
        CBR(base*4, bn=True, sample='up', activation='relu', dropout=False),
        CBR(base*2, bn=True, sample='up', activation='relu', dropout=False),
        CBR(base*1, bn=True, sample='up', activation='relu', dropout=False),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv1D(1, 3, strides=1, padding='same', kernel_initializer=initializer, activation='sigmoid')

    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])  # 最後のやつ以外足してんのか！
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)