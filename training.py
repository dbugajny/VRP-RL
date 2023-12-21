import numpy as np
import tensorflow as tf
from environment import Environment
from actor import Actor
from copy import deepcopy


def main():
    n_epochs = 100
    n_samples = 1
    n_locations = 10
    max_demand = 10
    max_capacity = 50
    big_number = 10000000

    actor = Actor(n_locations)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    env_org = Environment(n_samples, n_locations, max_demand, max_capacity)

    for _ in range(n_epochs):
        env = deepcopy(env_org)
        with tf.GradientTape(persistent=True) as tape:
            actions = []
            for _ in range(20):
                logits = actor(env, training=True) - env.mask * big_number

                logits_max = tf.nn.softmax(logits * 10)

                next_node = tf.reduce_mean(env.locations * tf.tile(tf.expand_dims(logits_max, -1), [1, 1, 2]), axis=1)

                env.update(tf.argmax(logits, 1))

                actions.append(next_node)  # because of softmax, next_node is not accurate

            acts = tf.convert_to_tensor(actions)  # shape [n_steps x n_samples x 2]
            acts_shifted = tf.concat((tf.expand_dims(actions[-1], 0), actions[:-1]), 0)

            distances = tf.math.sqrt(tf.reduce_sum(tf.math.square(acts_shifted - acts), -1) + 1e-12)
            summed_path = tf.reduce_sum(distances, axis=0)

            loss = tf.reduce_mean(summed_path)

        grads = tape.gradient(loss, actor.trainable_variables)
        grads_and_vars = zip(grads, actor.trainable_variables)
        optimizer.apply_gradients(grads_and_vars)

    print("TRAINING FINISHED ")


if __name__ == "__main__":
    main()
