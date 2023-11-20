import numpy as np
import tensorflow as tf
from state import State2
from actor import DummyActor, GreedyActor, GreedyActor2, ActorTF


def process_simulation():
    n_locations = 10
    temperature = 10
    state = State2(n_locations, 5, 10)
    actor = ActorTF(n_locations)
    optimizer = tf.keras.optimizers.Adam()

    with tf.GradientTape(persistent=True) as gradient_tape:
        # for i in range(1, 100):
        locations = state.locations.reshape(1, -1, 1)
        demands = state.demands.reshape(1, -1, 1).astype(float)
        vehicle_position = state.vehicle_position.reshape(1, -1, 1)
        current_capacity = np.array(state.current_capacity).reshape(1, -1, 1).astype(float)
        max_capacity = np.array(state.max_capacity).reshape(1, -1, 1).astype(float)

        logits = actor(locations, demands, vehicle_position, current_capacity, max_capacity)
        logits = logits / temperature
        next_location = np.argmax(logits.numpy()[0, 0])
        next_location = tf.random.categorical(logits, 1).numpy()[0, 0]

        state.update_state(next_location)

        loss_fn = tf.keras.losses.MeanSquaredError()
        loss = loss_fn(logits + next_location, tf.ones(logits.shape))

        # state.visualize_state()

        # if state.are_all_demands_satisfied():
        #     state.update_state(0)
        #     break

    gradients = gradient_tape.gradient(tf.constant(loss), actor.trainable_variables)
    print(gradients)
    grads_and_vars = zip(gradients, actor.trainable_variables)
    optimizer.apply_gradients(grads_and_vars)

    state.visualize_state()

if __name__ == "__main__":
    process_simulation()
