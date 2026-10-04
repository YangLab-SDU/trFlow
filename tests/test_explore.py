import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from trflow.explore import run_exploration


class ExplorationScheduleTests(unittest.TestCase):
    def test_seed_forward_uses_random_single_step_time(self):
        inf = SimpleNamespace(
            reprs={"pred_gemo": {"dist": torch.zeros(2, 2, 37)}},
            aatype=torch.zeros(2, dtype=torch.long),
        )
        output = {"cords_allatm": [torch.zeros(1, 1, 2, 3)]}
        core = SimpleNamespace(
            device=torch.device("cpu"),
            update_inf_strudata=Mock(return_value=inf),
            get_stru_repr=Mock(return_value=[output]),
        )
        pseudo_beta = torch.zeros(2, 3)
        new_dist_39 = torch.zeros(1, 2, 2, 39)

        with (
            patch(
                "trflow.explore.blend_distogram",
                return_value=(
                    inf.reprs["pred_gemo"]["dist"],
                    inf.reprs["pred_gemo"]["dist"].numpy(),
                    new_dist_39,
                    False,
                ),
            ),
            patch("trflow.explore.pseudo_beta_fn", return_value=pseudo_beta),
        ):
            run_exploration(core, inf, pseudo_beta, prior=object(), max_iters=1)

        call = core.update_inf_strudata.call_args
        self.assertEqual(call.kwargs["steps"], 1)
        self.assertFalse(call.kwargs["random_step"])
        self.assertTrue(call.kwargs["random_step_size"])
        core.get_stru_repr.assert_called_once_with(inf)


if __name__ == "__main__":
    unittest.main()
