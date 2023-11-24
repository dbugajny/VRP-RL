import tensorflow as tf
from environment import Environment

class Actor(tf.keras.Model):
    def __init__(self, n_locations):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(64)
        self.dense_2 = tf.keras.layers.Dense(64)
        self.dense_3 = tf.keras.layers.Dense(64)
        self.dense_4 = tf.keras.layers.Dense(n_locations)

    def call(self, environment: Environment):
        # query -> vehicle; ref -> locations
        demands = environment.demands
        capacity = environment.capacity

        x = self.dense_1(demands)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)

        return x
