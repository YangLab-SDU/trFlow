import unittest

import numpy as np

from trflow.config import RunOptions
from trflow.core import FlowInferenceCore


class FlowScheduleTests(unittest.TestCase):
    def test_single_step_uses_two_schedule_steps(self):
        self.assertEqual(RunOptions(single_step=True).effective_steps(), 2)

    def test_single_step_has_random_intermediate_noise_level(self):
        np.random.seed(7)
        first = FlowInferenceCore.make_schedule(2, random_step_size=True)
        second = FlowInferenceCore.make_schedule(2, random_step_size=True)

        self.assertEqual(first.shape, (3,))
        self.assertEqual(first[0], 1.0)
        self.assertEqual(first[-1], 0.0)
        self.assertGreater(first[1], 0.0)
        self.assertLess(first[1], 1.0)
        self.assertNotEqual(first[1], second[1])
        self.assertEqual(len(first[1:]) - 1, 1)

    def test_evenly_spaced_single_step_uses_midpoint(self):
        np.testing.assert_allclose(
            FlowInferenceCore.make_schedule(2, random_step_size=False),
            np.array([1.0, 0.5, 0.0]),
        )

    def test_degenerate_one_step_schedule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            FlowInferenceCore.make_schedule(1, random_step_size=True)


if __name__ == "__main__":
    unittest.main()
