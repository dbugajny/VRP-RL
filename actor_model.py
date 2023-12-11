import tensorflow as tf
from environment import Environment


class Actor(tf.keras.Model):
    def __init__(self, n_locations):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(1024, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(1024, activation="relu")
        self.dense_3 = tf.keras.layers.Dense(1024, activation="relu")
        self.dense_4 = tf.keras.layers.Dense(n_locations, activation="relu")

    def call(self, environment: Environment):
        # query -> vehicle; ref -> locations

        x1 = tf.keras.layers.Flatten()(environment.demands)
        x2 = tf.keras.layers.Flatten()(environment.capacity)
        x3 = tf.keras.layers.Flatten()(environment.locations)
        x4 = tf.keras.layers.Flatten()(environment.vehicle)

        x = tf.keras.layers.Concatenate()([x1, x2, x3, x4])
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)

        return x
