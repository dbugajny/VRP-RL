import matplotlib.pyplot as plt
import tensorflow as tf

BIG_NUMBER = 1000000


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


def run_environment_simulation(environment, actor, n_steps):
    approximated_actions = []
    real_actions = []
    for _ in range(n_steps):
        logits = actor(environment, training=True) - environment.mask * BIG_NUMBER

        logits_max = tf.nn.softmax(logits * 1)

        approximated_action = tf.reduce_mean(
            environment.locations * tf.tile(tf.expand_dims(logits_max, -1), [1, 1, 2]), axis=1
        )
        approximated_actions.append(approximated_action)

        environment.update(tf.argmax(logits, 1))
        real_actions.append(environment.vehicle)

    return approximated_actions, real_actions


def calculate_loss(actions):
    actions_shifted = tf.concat((tf.expand_dims(actions[-1], 0), actions[:-1]), 0)

    distances = tf.reduce_sum(tf.math.square(actions_shifted - actions), -1)  # (a^2 + b^2) instead of sqrt(a^2 + b^2)

    summed_path = tf.reduce_sum(distances, axis=0)

    loss = tf.reduce_mean(summed_path) * 10000

    return loss
