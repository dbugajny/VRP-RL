from utils import State
import tensorflow as tf


class Environment:
    def __init__(self, n_samples: int, n_locations: int, max_demand: int, max_capacity: int) -> None:
        self.n_locations = n_locations
        self.n_samples = n_samples

        self.locations = tf.random.uniform(
            shape=(n_samples, n_locations, 2), minval=-1, maxval=1
        )  # shape: [n_samples x n_locations x 2]

        self.demands = tf.cast(
            tf.random.uniform(shape=(n_samples, n_locations - 1), minval=1, maxval=max_demand, dtype=tf.int32),
            tf.float32,
        )
        self.demands = tf.concat(
            [tf.zeros(shape=(n_samples, 1)), self.demands], axis=1
        )  # shape: [n_samples x n_locations]

        self.mask = tf.zeros(shape=(n_samples, n_locations, 1))  # shape: [n_samples x n_locations]
        self.max_capacity = max_capacity
        self.capacity = tf.cast(tf.fill(dims=n_samples, value=max_capacity), tf.float32)  # shape: [n_samples]

    def update_state(self, next_node) -> State:
        range_idx = tf.expand_dims(tf.range(self.n_samples, dtype=tf.float32), 1)
        next_node_idx = tf.cast(
            tf.concat([range_idx, tf.expand_dims(tf.cast(next_node, tf.float32), -1)], 1), dtype=tf.int32
        )

        demand_satisfied = tf.minimum(tf.gather_nd(self.demands, next_node_idx), self.capacity)

        capacity_taken = tf.scatter_nd(next_node_idx, demand_satisfied, shape=tf.shape(self.demands))

        self.demands = tf.subtract(self.demands, capacity_taken)
        self.capacity -= demand_satisfied

        in_deopt_flag = tf.cast(tf.equal(next_node, 0), dtype=tf.float32)
        self.capacity = (
            tf.multiply(self.capacity, 1 - in_deopt_flag) + in_deopt_flag * self.max_capacity
        )

        # we don't want to go to places with mask == True
        self.mask = tf.concat([tf.zeros([self.n_samples, 1]), tf.cast(tf.equal(self.demands, 0), tf.float32)[:, 1:]], 1)

        self.mask += tf.concat(
            [
                # we don't want to stay in depot when there is still demand
                tf.expand_dims(
                    tf.multiply(
                        tf.cast(tf.greater(tf.reduce_sum(self.demands, 1), 0), tf.float32),  # there is still demand
                        tf.cast(tf.equal(next_node, 0), tf.float32),  # we are in depot
                    ),
                    -1,
                ),
                # when load == 0
                tf.tile(
                    tf.expand_dims(tf.cast(tf.equal(self.capacity, 0), tf.float32), -1),
                    [1, self.n_locations - 1],
                ),
            ],
            1,
        )

        return State(capacity=self.capacity, demands=self.demands, mask=self.mask)
