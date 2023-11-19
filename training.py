import numpy as np

from state import State
from actor import DummyActor, GreedyActor


def process_simulation():
    n_locations = 8
    state = State(n_locations, 5, 10)
    actor = GreedyActor()

    state.visualize_state()

    for i in range(1, 20):
        proba = actor(state)
        next_location = np.argmax(proba)
        state.update_state(next_location)

        state.visualize_state()

        if state.are_all_demands_satisfied():
            state.update_state(0)
            state.visualize_state()
            break


if __name__ == "__main__":
    process_simulation()
