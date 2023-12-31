import matplotlib.pyplot as plt
import tensorflow as tf

BIG_NUMBER = 10 ** 8
SMALL_NUMBER = 0.1 ** 8


def make_simulation_plot(locations, actions_lst):
    plt.figure(figsize=(8, 8))

    plt.scatter(locations[0, 0], locations[0, 1])
    plt.scatter(locations[1:, 0], locations[1:, 1])

    for i in range(len(actions_lst) - 1):
        plt.plot([actions_lst[i, 0], actions_lst[i + 1, 0]], [actions_lst[i, 1], actions_lst[i + 1, 1]])

    plt.plot([actions_lst[-1, 0], actions_lst[0, 0]], [actions_lst[-1, 1], actions_lst[0, 1]])

    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.show()


def run_environment_simulation(environment, actor, n_steps, approximation_level):
    approximated_actions = []
    real_actions = []
    for _ in range(n_steps):
        logits = actor(environment, training=True) - environment.mask * BIG_NUMBER

        logits_max = tf.nn.softmax(logits * approximation_level)

        approximated_action = tf.reduce_sum(
            environment.locations * tf.tile(tf.expand_dims(logits_max, -1), [1, 1, 2]), axis=1
        )
        approximated_actions.append(approximated_action)

        environment.update(tf.argmax(logits, 1))

        real_actions.append(environment.vehicle)

    return tf.convert_to_tensor(approximated_actions), tf.convert_to_tensor(real_actions)


def calculate_full_distance(actions):
    actions_shifted = tf.concat((tf.expand_dims(actions[-1], 0), actions[:-1]), 0)

    distances = tf.math.sqrt(tf.reduce_sum(tf.math.square(actions_shifted - actions), -1) + SMALL_NUMBER)

    full_distance = tf.reduce_sum(distances, axis=0)

    return full_distance
