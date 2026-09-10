"""Regression coverage for next-state pairing in D4RL exports."""

import unittest

import numpy as np

from dataset.D4RL import split_dataset_into_trajectories


class D4RLExportTests(unittest.TestCase):
    def make_dataset(self):
        return {
            "observations": np.array([[0], [1], [10], [11], [20], [21], [22]]),
            "actions": np.arange(7).reshape(-1, 1),
            "rewards": np.arange(7),
            "terminals": np.array([0, 1, 0, 0, 0, 0, 0]),
            "timeouts": np.array([0, 0, 0, 1, 0, 0, 0]),
        }

    def test_supplied_next_states_keep_boundaries_and_incomplete_tail(self):
        dataset = self.make_dataset()
        dataset["next_observations"] = dataset["observations"] + 0.5
        for timeout_as_done in (False, True):
            with self.subTest(timeout_as_done=timeout_as_done):
                trajs = split_dataset_into_trajectories(dataset, timeout_as_done)
                self.assertEqual(trajs["lengths"], [2, 2, 3])
                for output_key, input_key in (
                    ("states", "observations"),
                    ("next_states", "next_observations"),
                    ("actions", "actions"),
                    ("rewards", "rewards"),
                ):
                    np.testing.assert_array_equal(
                        np.concatenate(trajs[output_key]), dataset[input_key]
                    )
                np.testing.assert_array_equal(
                    np.concatenate(trajs["dones"]),
                    [0, 1, 0, float(timeout_as_done), 0, 0, 0],
                )

    def test_inferred_next_states_never_cross_boundaries_or_repeat_last_row(self):
        for timeout_as_done in (False, True):
            with self.subTest(timeout_as_done=timeout_as_done):
                trajs = split_dataset_into_trajectories(
                    self.make_dataset(), timeout_as_done
                )
                self.assertEqual(trajs["lengths"], [1, 1, 2])
                np.testing.assert_array_equal(
                    np.concatenate(trajs["states"]), [[0], [10], [20], [21]]
                )
                np.testing.assert_array_equal(
                    np.concatenate(trajs["next_states"]), [[1], [11], [21], [22]]
                )
                np.testing.assert_array_equal(
                    np.concatenate(trajs["actions"]), [[0], [2], [4], [5]]
                )
                np.testing.assert_array_equal(
                    np.concatenate(trajs["rewards"]), [0, 2, 4, 5]
                )
                np.testing.assert_array_equal(np.concatenate(trajs["dones"]), 0)

    def test_consecutive_boundaries_do_not_create_empty_trajectories(self):
        dataset = self.make_dataset()
        dataset["terminals"] = np.array([1, 1, 0, 0, 0, 0, 1])
        trajs = split_dataset_into_trajectories(dataset)
        self.assertEqual(trajs["lengths"], [1, 2])
        np.testing.assert_array_equal(
            np.concatenate(trajs["states"]), [[10], [20], [21]]
        )

    def test_missing_timeouts_and_short_datasets(self):
        for size in (0, 1, 2):
            for supplied in (False, True):
                with self.subTest(size=size, supplied=supplied):
                    dataset = {key: value[:size] for key, value in self.make_dataset().items()}
                    del dataset["timeouts"]
                    if supplied:
                        dataset["next_observations"] = dataset["observations"] + 0.5
                    trajs = split_dataset_into_trajectories(dataset)
                    expected_length = size if supplied else max(0, size - 1)
                    self.assertEqual(
                        trajs.get("lengths", []),
                        [expected_length] if expected_length else [],
                    )
                    for key in ("states", "next_states", "actions", "rewards", "dones"):
                        for trajectory in trajs.get(key, []):
                            self.assertEqual(trajectory.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
