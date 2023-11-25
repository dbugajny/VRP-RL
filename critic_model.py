import tensorflow as tf


class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(64)
        self.dense_2 = tf.keras.layers.Dense(64)
        self.dense_3 = tf.keras.layers.Dense(64)
        self.dense_4 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)

        return x
