from utils import calculate_full_distance
import numpy as np


def test_calculate_full_distance_1_sample_2_dim():
    data = np.array([[1, 1], [2, 2], [4, 4], [0, 0]], dtype=np.float32).reshape(-1, 1, 2)
    result = calculate_full_distance(data).numpy()

    expected = np.sqrt(2) * 8

    assert np.abs(result - expected) < 0.00001


def test_calculate_full_distance_3_samples_1_dim():
    data = np.array(
        [
            [[7], [5], [6]],  # move 1
            [[2], [3], [9]],  # move 2
            [[4], [4], [5]],  # move 3
        ],
        dtype=np.float32,
    )

    result = calculate_full_distance(data).numpy()

    expected = np.array([10, 4, 8])

    assert np.max(np.abs(result - expected)) < 0.00001
