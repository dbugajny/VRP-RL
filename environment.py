import tensorflow as tf


class Environment:
    def __init__(self, n_samples: int, n_locations: int, max_demand: int, max_capacity: int) -> None:
        self.n_locations = n_locations
        self.n_samples = n_samples

        self.locations = tf.random.uniform(
            shape=(n_samples, n_locations, 2), minval=-1000, maxval=1000
        )  # shape: [n_samples x n_locations x 2]
        self.vehicle = self.locations[:, 0, :]  # shape: [n_samples x 2]

        self.demands = tf.concat(
            [
                tf.zeros(shape=(n_samples, 1)),  # depot
                tf.cast(
                    tf.random.uniform(shape=(n_samples, n_locations - 1), minval=1, maxval=max_demand, dtype=tf.int32),
                    tf.float32,
                ),  # customers
            ],
            axis=1,
        )  # shape: [n_samples x n_locations]

        self.mask = tf.concat(
            [tf.ones(shape=(n_samples, 1)), tf.zeros(shape=(n_samples, n_locations - 1))], 1  # depot  # locations
        )  # shape: [n_samples x n_locations]

        self.max_capacity = max_capacity
        self.capacity = tf.cast(tf.fill(dims=n_samples, value=self.max_capacity), tf.float32)  # shape: [n_samples]

    def update(self, next_node) -> None:
        """Process one step:
            1. Calculate how many goods were taken by the vehicle.
            2. Update demands and capacity.
            3. Create mask - vechicle won't go to nodes with mask == True
        """
        range_idx = tf.expand_dims(tf.range(self.n_samples, dtype=tf.float32), 1)
        next_node_idx = tf.cast(
            tf.concat([range_idx, tf.reshape(tf.cast(next_node, dtype=tf.float32), shape=[-1, 1])], 1), dtype=tf.int32
        )
        self.vehicle = tf.gather_nd(self.locations, next_node_idx)

        demand_satisfied = tf.minimum(tf.gather_nd(self.demands, next_node_idx), self.capacity)

        capacity_taken = tf.scatter_nd(next_node_idx, demand_satisfied, shape=tf.shape(self.demands))

        self.demands = tf.subtract(self.demands, capacity_taken)
        self.capacity = tf.subtract(self.capacity, demand_satisfied)

        in_depot_flag = tf.cast(tf.equal(next_node, 0), dtype=tf.float32)
        self.capacity = tf.multiply(self.capacity, 1 - in_depot_flag) + in_depot_flag * self.max_capacity

        # we don't want to go to places with demand == 0
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
                # when capacity == 0
                tf.tile(
                    tf.expand_dims(tf.cast(tf.equal(self.capacity, 0), tf.float32), -1),
                    [1, self.n_locations - 1],
                ),
            ],
            1,
        )
