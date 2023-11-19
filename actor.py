import tensorflow as tf
import numpy as np
from state import State


class Actor(tf.keras.Model):
    def __init__(self):
        pass

    def call(self):
        pass


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

