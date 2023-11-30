import numpy as np
import tensorflow as tf
from environment import Environment
from actor_model import Actor
from copy import deepcopy

def main():
    n_samples = 10000
    n_locations = 10
    max_demand = 10
    max_capacity = 5

    actor = Actor(n_locations)
    optimizer = tf.keras.optimizers.Adam()

    env_org = Environment(n_samples, n_locations, max_demand, max_capacity)

    for i in range(100):
        env = deepcopy(env_org)
        with tf.GradientTape(persistent=True) as tape:
            actions = []
            for i in range(20):
                logits = actor(env, training=True) - env.mask * 1000000

                logits_max = tf.nn.softmax(logits * 100)

                action = tf.reduce_sum(env.locations * tf.tile(tf.expand_dims(logits_max, -1), [1, 1, 2]), axis=1)

                env.update(tf.argmax(logits, 1))

                actions.append(action)

            acts = tf.convert_to_tensor(actions)  # shape [n_steps x n_samples x 2]
            acts_shifted = tf.concat((tf.expand_dims(actions[-1], 0), actions[:-1]), 0)

            distances = tf.math.sqrt(tf.reduce_sum(tf.math.square(acts_shifted - acts), -1) + 1e-12)
            summed_path = tf.reduce_sum(distances, axis=0)

            loss = tf.reduce_mean(summed_path * 10000000)
            print(loss)

        grads = tape.gradient(loss, actor.trainable_variables)
        grads_and_vars = zip(grads, actor.trainable_variables)
        optimizer.apply_gradients(grads_and_vars)

    print("FINISHED ")


if __name__ == "__main__":
    main()
