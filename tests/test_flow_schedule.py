import unittest

import numpy as np

from trflow.config import RunOptions
from trflow.core import FlowInferenceCore


class FlowScheduleTests(unittest.TestCase):
    def test_single_step_uses_one_model_forward(self):
        self.assertEqual(RunOptions(single_step=True).effective_steps(), 1)

    def test_single_step_has_random_intermediate_noise_level(self):
        np.random.seed(7)
        first = FlowInferenceCore.make_schedule(1, random_step_size=True)
        second = FlowInferenceCore.make_schedule(1, random_step_size=True)

        self.assertEqual(first.shape, (3,))
        self.assertEqual(first[0], 1.0)
        self.assertEqual(first[-1], 0.0)
        self.assertGreater(first[1], 0.0)
        self.assertLess(first[1], 1.0)
        self.assertNotEqual(first[1], second[1])
        self.assertEqual(len(first[1:]) - 1, 1)

    def test_evenly_spaced_single_step_uses_midpoint(self):
        np.testing.assert_allclose(
            FlowInferenceCore.make_schedule(1, random_step_size=False),
            np.array([1.0, 0.5, 0.0]),
        )

    def test_seven_steps_produce_seven_model_forwards(self):
        schedule = FlowInferenceCore.make_schedule(7, random_step_size=True)
        self.assertEqual(schedule.shape, (9,))
        self.assertEqual(len(schedule[1:]) - 1, 7)

    def test_zero_model_forwards_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            FlowInferenceCore.make_schedule(0, random_step_size=True)


if __name__ == "__main__":
    unittest.main()
