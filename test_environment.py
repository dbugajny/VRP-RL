from environment import Environment
import tensorflow as tf


def test_run_environment_simulation():
    env = Environment(2, 3, 3, 2)

    env.locations = tf.convert_to_tensor(
        [
            [[1, 1], [2, 2], [3, 3]],
            [[-1, -1], [-2, -2], [-3, -3]],
        ],
        dtype=tf.float32,
    )

    env.demands = tf.convert_to_tensor([[0, 3, 2], [0, 1, 1]], dtype=tf.float32)

    env.vehicle = env.locations[:, 0, :]

    next_node = tf.convert_to_tensor([1, 2], dtype=tf.float32)

    env.update(next_node)

    assert tf.reduce_all(env.vehicle == tf.convert_to_tensor([[2, 2], [-3, -3]], dtype=tf.float32))
    assert tf.reduce_all(env.capacity == tf.convert_to_tensor([0, 1], dtype=tf.float32))
    assert tf.reduce_all(env.mask == tf.convert_to_tensor([[0, 1, 1], [0, 0, 1]], dtype=tf.float32))
    assert tf.reduce_all(env.demands == tf.convert_to_tensor([[0, 1, 2], [0, 1, 0]], dtype=tf.float32))
