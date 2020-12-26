import tensorflow as tf


##################################################################################
# Loss function
##################################################################################

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def discriminator_loss(real, fake):
    loss = []
    for i in range(len(fake)):
        real_loss = tf.reduce_mean(tf.math.squared_difference(real[i][-1], 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake[i][-1]))
        loss.append(real_loss + fake_loss)

    return tf.reduce_mean(loss)

def generator_loss(fake):
    loss = []

    for i in range(len(fake)):
        fake_loss = tf.reduce_mean(tf.math.squared_difference(fake[i][-1], 1.0))
        loss.append(fake_loss)

    return tf.reduce_mean(loss)

def feature_loss(real, fake) :

    loss = []

    for i in range(len(fake)) :
        intermediate_loss = 0
        for j in range(len(fake[i]) - 1) :
            intermediate_loss += L1_loss(real[i][j], fake[i][j])
        loss.append(intermediate_loss)

    return tf.reduce_mean(loss)

def z_sample(mean, logvar):
    eps = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)
    return mean + tf.exp(logvar * 0.5) * eps

def kl_loss(mean, logvar):
    # shape : [batch_size, channel]
    loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - 1 - logvar)
    # loss = tf.reduce_mean(loss)

    return loss