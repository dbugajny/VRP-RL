import tensorflow as tf
from environment import Environment


class Critic(tf.keras.Model):
    ''' Basically same network as actor, but with single output neuron 
        and a task to estimate the quality of the action (loss is calculated differently)
    '''
    def __init__(self):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(64)
        self.dense_2 = tf.keras.layers.Dense(64)
        self.dense_3 = tf.keras.layers.Dense(64)
        self.dense_4 = tf.keras.layers.Dense(1)

    def call(self, environment: Environment, tour):
        x1 = tf.keras.layers.Flatten()(environment.demands)
        x2 = tf.keras.layers.Flatten()(environment.capacity)
        x3 = tf.keras.layers.Flatten()(environment.locations)
        x4 = tf.keras.layers.Flatten()(environment.vehicle)
        x5 = tf.keras.layers.Flatten()(tour)
        
        x = tf.keras.layers.Concatenate()([x1, x2, x3, x4, x5])
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = self.dense_1(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        x = self.dense_2(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        x = self.dense_3(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        x = self.dense_4(x)
        return x
