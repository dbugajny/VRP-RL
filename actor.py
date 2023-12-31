import tensorflow as tf
from environment import Environment


class DenseActor(tf.keras.Model):
    def __init__(self, n_locations):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(1024, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(1024, activation="relu")
        self.dense_3 = tf.keras.layers.Dense(1024, activation="relu")
        self.dense_4 = tf.keras.layers.Dense(n_locations)

    def call(self, environment: Environment):
        # query -> vehicle; ref -> locations

        x1 = tf.keras.layers.Flatten()(environment.demands)
        x2 = tf.keras.layers.Flatten()(environment.capacity)
        x3 = tf.keras.layers.Flatten()(environment.locations)
        x4 = tf.keras.layers.Flatten()(environment.vehicle)

        x = tf.keras.layers.Concatenate()([x1, x2, x3, x4])

        x = self.dense_1(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        # x = tf.keras.layers.Dropout(0.1)(x)

        x = self.dense_2(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        # x = tf.keras.layers.Dropout(0.1)(x)

        x = self.dense_3(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        # x = tf.keras.layers.Dropout(0.1)(x)

        x = self.dense_4(x)

        return x


class RandomActor(tf.keras.Model):
    def __init__(self, n_locations):
        super().__init__()
        self.n_locations = n_locations

    def call(self, environment: Environment):
        return tf.random.normal(shape=(environment.locations.shape[0], self.n_locations))
