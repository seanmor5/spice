class UNetEncoderBlock(tf.keras.layers.Layer):

	def __init__(self, filters, size, apply_batchnorm=True, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02)):
    super(__init__, UNetEncoderBlock).__init__()

    self.initializer = initializer

    self.conv = tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=self.initializer)

    self.apply_batchnorm = apply_batchnorm

    if apply_batchnorm:
      self.batchnorm = tf.keras.layers.BatchNormalization()

    self.relu = tf.keras.layers.LeakyReLU()

  def call(self, x):
    x = self.conv(x)
    x = self.batchnorm(x) if self.apply_batchnorm else x
    x = self.relu(x)
    return x

class UNetDecoderBlock(tf.keras.layers.Layer):

  def __init__(self, filters, size, apply_dropout=True, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02)):
    super(__init__, UNetDecoderBlock).__init__()

    self.initializer = initializer
    self.apply_dropout = apply_dropout

    self.conv = tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=self.initializer)

    self.batchnorm = tf.keras.layers.BatchNormalization()

    if self.apply_dropout:
      self.dropout = tf.keras.layers.Dropout(0.5)

    self.relu = tf.keras.layers.ReLU()

  def call(self, x):
    x = self.conv(x)
    x = self.batchnorm(x)
    x = self.dropout(x) if self.apply_dropout else x
    x = self.relu(x)
    return x