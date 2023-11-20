import tensorflow as tf
import numpy as np
from state import State, State2


class ActorTF(tf.keras.Model):
    def __init__(self, n_locations):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(64)
        self.dense_2 = tf.keras.layers.Dense(64)
        self.dense_3 = tf.keras.layers.Dense(64)
        self.dense_4 = tf.keras.layers.Dense(n_locations)

    def call(self, locations, demands, vehicle_position, current_capacity, max_capacity):
        x = tf.keras.layers.Concatenate(axis=1)([locations, demands, vehicle_position, current_capacity, max_capacity])
        x = tf.keras.layers.Flatten()(x)

        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)

        return x


class DummyActor:
    def __init__(self):
        pass

    def __call__(self, state: State, *args, **kwargs):
        return np.random.rand(len(state.locations))


class GreedyActor:
    def __init__(self):
        pass

    def __call__(self, state: State, *args, **kwargs):
        if state.vehicle.current_capacity == 0:
            proba = np.zeros(shape=(len(state.locations)))
            proba[0] = 1
            return proba

        distances = [
            np.sqrt(
                (state.vehicle.coordinates.x - location.coordinates.x) ** 2
                + (state.vehicle.coordinates.y - location.coordinates.y) ** 2
            )
            for location in state.locations
        ]

        best_idx = None
        best_distance = float("inf")

        for i in range(len(state.locations)):
            if state.locations[i].demand == 0 or state.locations[i].is_depot:
                continue
            if best_distance > distances[i]:
                best_distance = distances[i]
                best_idx = i

        proba = np.zeros(shape=(len(state.locations)))
        proba[best_idx] = 1

        return proba


class GreedyActor2:
    def __init__(self):
        pass

    def __call__(self, state: State2, *args, **kwargs):
        if state.current_capacity == 0:
            proba = np.zeros(shape=(len(state.locations)))
            proba[0] = 1
            return proba

        distances = [
            np.sqrt(
                (state.vehicle_position[0] - location[0]) ** 2
                + (state.vehicle_position[1] - location[1]) ** 2
            )
            for location in state.locations
        ]

        best_idx = None
        best_distance = float("inf")

        for i in range(1, len(state.demands)):
            if state.demands[i] == 0:
                continue
            if best_distance > distances[i]:
                best_distance = distances[i]
                best_idx = i

        proba = np.zeros(shape=(len(state.locations)))
        proba[best_idx] = 1

        return proba
