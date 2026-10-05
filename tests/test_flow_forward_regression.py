"""Exercise the real sampling loop without model weights or a GPU."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from trflow.core import FlowInferenceCore, Inf_StruData


class _RecordingStructureModel:
    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def __call__(self, **kwargs):
        coords = kwargs["noisy_cb"].clone()
        self.calls.append((float(kwargs["t_step"][0]), coords))
        # The real pseudo-beta extractor sees atom14 coordinates [1, 14, L, 3].
        return {"cords_allatm": [coords[None, None].repeat(1, 14, 1, 1)]}


class _FixedPrior:
    def __init__(self):
        self.calls = 0
        # Non-rigidly related to the reference so alignment cannot erase noise.
        self.coords = torch.tensor(
            [[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 6.0]]
        )

    def sample(self):
        self.calls += 1
        return self.coords.clone()


class FlowForwardRegressionTests(unittest.TestCase):
    def setUp(self):
        self.reference = torch.tensor(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
        )
        self.numpy_state = np.random.get_state()

    def tearDown(self):
        np.random.set_state(self.numpy_state)

    def _infer(self, steps, random_step_size=False, random_step=False):
        core = FlowInferenceCore.__new__(FlowInferenceCore)
        core.device = torch.device("cpu")
        core.structure_model = _RecordingStructureModel()
        prior = _FixedPrior()
        inf = Inf_StruData(
            raw_seq="AAAA",
            reprs={"reprs": {}},
            pred_gemo={},
            cb_mask=torch.ones(1, 4),
            aatype=torch.zeros(4, dtype=torch.long),
        )
        core.update_inf_strudata(
            inf,
            self.reference,
            prior,
            steps=steps,
            random_step=random_step,
            random_step_size=random_step_size,
        )
        outputs = core.get_stru_repr(inf)
        return inf, prior, core.structure_model.calls, outputs

    def test_actual_forward_counts_and_initial_noise(self):
        for steps in (1, 2, 7):
            with self.subTest(steps=steps):
                inf, prior, calls, outputs = self._infer(steps)
                self.assertEqual(len(calls), steps)
                self.assertEqual(len(outputs), steps)
                self.assertEqual(prior.calls, 1)
                np.testing.assert_allclose(
                    [time for time, _coords in calls], inf.schedule[:-1], rtol=1e-6
                )
                self.assertTrue(all(0 < time < 1 for time, _coords in calls))
                start = inf.schedule[0]
                expected_input = start * inf.noisy + (1 - start) * self.reference
                self.assertTrue(torch.allclose(calls[0][1], expected_input, atol=1e-6))
                self.assertFalse(torch.allclose(calls[0][1], self.reference))

    def test_random_single_forward_has_coordinate_diversity(self):
        np.random.seed(17)
        _inf, _prior, first_calls, first_outputs = self._infer(1, random_step_size=True)
        _inf, _prior, second_calls, second_outputs = self._infer(1, random_step_size=True)
        self.assertEqual(len(first_calls), 1)
        self.assertEqual(len(second_calls), 1)
        self.assertNotEqual(first_calls[0][0], second_calls[0][0])
        self.assertFalse(torch.allclose(first_calls[0][1], second_calls[0][1]))
        self.assertFalse(
            torch.allclose(
                first_outputs[0]["cords_allatm"][-1],
                second_outputs[0]["cords_allatm"][-1],
            )
        )

    def test_random_single_forward_is_reproducible_with_the_same_seed(self):
        np.random.seed(29)
        _inf, _prior, first_calls, _outputs = self._infer(1, random_step_size=True)
        np.random.seed(29)
        _inf, _prior, second_calls, _outputs = self._infer(1, random_step_size=True)
        self.assertEqual(first_calls[0][0], second_calls[0][0])
        self.assertTrue(torch.equal(first_calls[0][1], second_calls[0][1]))

    def test_random_forward_count_executes_each_supported_choice(self):
        for chosen_steps in (1, 7):
            with self.subTest(chosen_steps=chosen_steps):
                with patch("trflow.core.random.choice", return_value=chosen_steps) as choose:
                    _inf, _prior, calls, outputs = self._infer(3, random_step=True)
                choose.assert_called_once_with([1, 7])
                self.assertEqual(len(calls), chosen_steps)
                self.assertEqual(len(outputs), chosen_steps)


if __name__ == "__main__":
    unittest.main()
