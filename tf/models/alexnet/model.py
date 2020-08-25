class AlexNet(tf.keras.Model):

  def __init__(self, depth_radius=5, bias=2, alpha=1e-4, beta=0.75):
    super(AlexNet, self).__init__()

    self.depth_radius = 5
    self.bias = 2
    self.alpha = alpha
    self.beta = beta
    self.initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)
    self.bias_initializer = tf.keras.initializers.Ones()

    self.conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', padding='same', kernel_initializer=self.initializer, name='alex_conv1')
    self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2) name='alex_pool1')

    self.conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer=self.initializer, bias_initializer=self.bias_initializer, name='alex_conv2')
    self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='alex_pool2')

    self.conv3 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer=self.initializer, bias_initializer=self.bias_initializer, name='alex_conv3')

    self.conv4 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer=self.initializer, bias_initializer=self.bias_initializer, name='alex_conv4')

    self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer=self.initializer, bias_initializer=self.bias_initializer, name='alex_conv5')
    self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='alex_pool3')

    self.flatten = tf.keras.layers.Flatten()

    self.fc1 = tf.keras.layers.Dense(4096, activation='relu', kernel_initializer=self.initializer, bias_initializer=self.bias_initializer, name='alex_fc1')
    self.dropout1 = tf.keras.layers.Dropout(0.5)

    self.fc2 = tf.keras.layers.Dense(4096, activation='relu', kernel_initializer=self.initializer, bias_initializer=self.bias_initializer, name='alex_fc2')
    self.dropout2 = tf.keras.layers.Dropout(0.5)

    self.classifier = tf.keras.layers.Dense(1000, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = tf.nn.local_response_normalization(x, depth_radius=self.depth_radius, bias=self.bias, alpha=self.alpha, beta=self.beta, name='local_response_norm1')
    x = self.pool1(x)
    x = self.conv2(x)
    x = tf.nn.local_response_normalization(x, depth_radius=self.depth_radius, bias=self.bias, alpha=self.alpha, beta=self.beta, name='local_response_norm2')
    x = self.pool2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.pool3(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.dropout1(x)
    x = self.fc2(x)
    x = self.dropout2(x)
    return self.classifier(x)
