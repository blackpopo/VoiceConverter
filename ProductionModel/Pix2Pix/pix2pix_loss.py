import tensorflow as tf
from config import Pix2PixConfig

config = Pix2PixConfig()

def generator_loss(loss_object, disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(gen_output - target))
  #hinge loss


  #This loss is largely affected by soucrce
  # msle = tf.keras.losses.msle(gen_output, target)


  total_gen_loss = (config.ADV * gan_loss) + (config.LAMBDA * loss)

  return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(loss_object, disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

def generator_loss2(gen_output, target):
  shape = gen_output.shape
  if len(shape) == 4:
    pre = tf.slice(gen_output, (0, 0, 0, 0), (shape[0], shape[1]-1, shape[2], shape[3]))
    post = tf.slice(gen_output, (0, 1, 0, 0), (shape[0], shape[1]-1, shape[2], shape[3]))
  elif len(shape) == 3:
    pre = tf.slice(gen_output, (0, 0, 0), (shape[0], shape[1]-1, shape[2]))
    post = tf.slice(gen_output, (0, 1, 0), (shape[0], shape[1]-1, shape[2]))
  else:
    pre = 0
    post = 0

 #まったく使えんｗｗｗ
  # hinge_loss = tf.reduce_mean(tf.keras.losses.hinge(gen_output, target))

#mselと同じくらい
  mse_loss = tf.reduce_mean(tf.square(gen_output - target))

#結構いい！
  msel_loss = tf.reduce_mean(tf.keras.losses.msle(gen_output, target))

  gen_time_loss = tf.reduce_mean(tf.abs(pre-post))

  gen_l1_loss = tf.reduce_mean(tf.abs(gen_output - target))

  # huber = tf.reduce_mean(tf.keras.losses.logcosh(gen_output, target))

  return gen_time_loss, mse_loss