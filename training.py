import numpy as np
import tensorflow as tf
from environment import Environment
from actor import Actor


def main():
    n_samples = 5
    n_locations = 10
    max_demand = 10
    max_capacity = 20
    env = Environment(n_samples, n_locations, max_demand, max_capacity)
    act = Actor(n_locations)

    with tf.GradientTape(persistent=True) as tape:
        actions = []
        for i in range(20):
            logits = act(env, training=True) - env.mask * 1000000

            logits_max = tf.nn.softmax(logits * 100)

            action = tf.reduce_sum(env.locations * tf.tile(tf.expand_dims(logits_max, -1), [1, 1, 2]), axis=1)

            env.update(tf.argmax(logits, 1))

            actions.append(action)

        acts = tf.convert_to_tensor(actions)  # shape [n_steps x n_samples x 2]
        acts_2 = tf.concat((tf.expand_dims(actions[-1], 0), actions[:-1]), 0)

        distances = tf.math.sqrt(tf.reduce_sum(tf.math.square(acts_2 - acts), -1) + 1e-12)
        summed_path = tf.reduce_sum(distances, axis=0)

        loss = tf.reduce_mean(summed_path * 10000000)

    grads = tape.gradient(loss, act.trainable_variables)


if __name__ == "__main__":
    main()
