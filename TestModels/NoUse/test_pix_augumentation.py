import tensorflow as tf
from test_config import Pix2PixConfig
config = Pix2PixConfig()

# normalizing the images to [-1, 1]

# def normalize(input_image, real_image):
#   input_image = (input_image / 127.5) - 1
#   real_image = (real_image / 127.5) - 1
#
#   return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
  # resizing to 286 x 286 x 3
  input_image, real_image = resize(input_image, real_image, 572, 572)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, config.IMG_HEIGHT, config.IMG_WIDTH, 1])

  return cropped_image[0], cropped_image[1]

def preprocess(data):
  input_image, real_image = data[0], data[1]
  # input_image, real_image = random_jitter(input_image, real_image)
  return input_image, real_image

def newaxis(data):
  input_image, real_image = data[0], data[1]
  return input_image, real_image

