import tensorflow as tf


class ActorTF(tf.keras.Model):
    def __init__(self, n_locations):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(64)
        self.dense_2 = tf.keras.layers.Dense(64)
        self.dense_3 = tf.keras.layers.Dense(64)
        self.dense_4 = tf.keras.layers.Dense(n_locations)

    def call(self, locations, demands, vehicle_position, current_capacity, max_capacity, l):
        x = tf.keras.layers.Concatenate(axis=1)([locations, demands, vehicle_position, current_capacity, max_capacity])
        x = tf.keras.layers.Flatten()(x)

        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)

        next_location = tf.random.categorical(x, 1).numpy()[0, 0]

        return x, l[next_location], next_location

