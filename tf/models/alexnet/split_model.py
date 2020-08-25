class AlexNet(tf.keras.Model):

  def __init__(self, n=5, k=2, alpha=1e-4, beta=0.75):
    super(AlexNet, self).__init__()

    # Hyperparameters
    self.depth_radius = n
    self.bias = k
    self.alpha = alpha
    self.beta = beta

    # Initializer
    self.initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)

    # GPU 0 blocks
    with tf.device('GPU:0'):
      self.conv0_1 = tf.keras.layers.Conv2D(48, (11, 11), strides=(4, 4), activation='relu', padding='same', kernel_initializer=self.initializer, name='alex_conv0_1')
      self.pool0_1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='alex_pool0_1')
      self.conv0_2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer=self.initializer, name='alex_conv0_2')
      self.pool0_2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='alex_pool0_2')

      self.conv0_3 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer=self.initializer, name='alex_conv0_3')
      self.conv0_4 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer=self.initializer, name='alex_conv0_4')
      self.conv0_5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer=self.initializer, name='alex_conv0_5')
      self.pool0_3 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='alex_pool1_3')

      self.flatten0 = tf.keras.layers.Flatten()

      self.fc0_1 = tf.keras.layers.Dense(2048, activation='relu', kernel_initializer=self.initializer, name='alex_fc0_1')
      self.dropout0_1 = tf.keras.layers.Dropout(0.5)

      self.fc0_2 = tf.keras.layers.Dense(2048, activation='relu', kernel_initializer=self.initializer, name='alex_fc0_2')
      self.dropout0_2 = tf.keras.layers.Dropout(0.5)

    # GPU 1 blocks
    with tf.device('GPU:1'):
      self.conv1_1 = tf.keras.layers.Conv2D(filters=48, kernel_size=(11, 11), strides=(4, 4), activation='relu', padding='same', kernel_initializer=self.initializer, name='alex_conv1_1')
      self.pool1_1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='alex_pool1_1')
      self.conv1_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer=self.initializer, name='alex_conv1_2')
      self.pool1_2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='alex_pool1_2')

      self.conv1_3 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer=self.initializer, name='alex_conv1_3')
      self.conv1_4 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer=self.initializer, name='alex_conv1_4')
      self.conv1_5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer=self.initializer, name='alex_conv1_5')
      self.pool1_3 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='alex_pool1_3')

      self.flatten1 = tf.keras.layers.Flatten()

      self.fc1_1 = tf.keras.layers.Dense(2048, activation='relu', kernel_initializer=self.initializer, name='alex_fc1_1')
      self.dropout1_1 = tf.keras.layers.Dropout(0.5)

      self.fc1_2 = tf.keras.layers.Dense(2048, activation='relu', kernel_initializer=self.initializer, name='alex_fc1_2')
      self.dropout1_2 = tf.keras.layers.Dropout(0.5)

      self.classifier = tf.keras.layers.Dense(1000, activation='softmax', kernel_initializer=self.initializer, name='alex_classifier')

  def call(self, inputs):

    x_0 = self.conv0_1(inputs)
    x_0 = tf.nn.local_response_normalization(x_0, depth_radius=self.depth_radius, bias=self.bias, alpha=self.alpha, beta=self.beta)
    x_0 = self.pool0_1(x_0)

    y_0 = self.conv1_1(inputs)
    y_0 = tf.nn.local_response_normalization(y_0, depth_radius=self.depth_radius, bias=self.bias, alpha=self.alpha, beta=self.beta)
    y_0 = self.pool1_1(y_0)

    x_0 = self.conv0_2(x_0)
    x_0 = tf.nn.local_response_normalization(x_0, depth_radius=self.depth_radius, bias=self.bias, alpha=self.alpha, beta=self.beta)
    x_0 = self.pool0_2(x_0)

    y_0 = self.conv1_2(y_0)
    y_0 = tf.nn.local_response_normalization(y_0, depth_radius=self.depth_radius, bias=self.bias, alpha=self.alpha, beta=self.beta)
    y_0 = self.pool1_2(y_0)

    x_1 = tf.concat([x_0, y_0], axis=0)
    x_1 = self.conv0_3(x_1)
    x_1 = self.conv0_4(x_1)
    x_1 = self.conv0_5(x_1)
    x_1 = self.pool0_3(x_1)
    x_1 = self.flatten0(x_1)

    y_1 = tf.concat([y_0, x_0], axis=0)
    y_1 = self.conv1_3(y_1)
    y_1 = self.conv1_4(y_1)
    y_1 = self.conv1_5(y_1)
    y_1 = self.pool1_3(y_1)
    y_1 = self.flatten1(y_1)

    x_2 = tf.concat([x_1, y_1], axis=-1)
    x_2 = self.fc0_1(x_2)
    x_2 = self.dropout0_1(x_2)

    y_2 = tf.concat([y_1, x_1], axis=-1)
    y_2 = self.fc1_1(y_2)
    y_2 = tf.keras.layers.Dropout(0.5)(y_2)

    x_3 = tf.concat([x_2, y_2], axis=-1)
    x_3 = self.fc0_2(x_3)
    x_3 = self.dropout0_2(x_3)

    y_3 = tf.concat([y_2, x_2], axis=-1)
    y_3 = self.fc1_2(y_3)
    y_3 = self.dropout1_2(y_3)

    out = tf.concat([x_3, y_3], axis=-1)
    return self.classifier(out)