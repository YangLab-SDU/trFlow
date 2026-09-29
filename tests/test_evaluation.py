import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trflow import evaluation


class EvaluationTests(unittest.TestCase):
    def test_default_tmscore_is_bundled_executable(self):
        self.assertEqual(
            Path(evaluation.find_tmscore(None)),
            evaluation.DEFAULT_TMSCORE.resolve(),
        )

    def test_explicit_tmscore_path_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "TMscore"
            executable.touch()
            executable.chmod(executable.stat().st_mode | 0o111)
            self.assertEqual(
                Path(evaluation.find_tmscore(str(executable))),
                executable.resolve(),
            )

    def test_score_pair_always_passes_seq(self):
        output = (
            "RMSD of  the common residues=    1.234\n"
            "TM-score    = 0.5678  (d0= 4.26)\n"
        )
        completed = type(
            "Completed",
            (),
            {"returncode": 0, "stdout": output, "stderr": ""},
        )()
        with patch(
            "trflow.evaluation.subprocess.run",
            return_value=completed,
        ) as run:
            self.assertEqual(
                evaluation.score_pair(
                    "/tmp/TMscore",
                    Path("pred.pdb"),
                    Path("native.pdb"),
                ),
                (1.234, 0.5678),
            )
        self.assertEqual(
            run.call_args.args[0],
            ["/tmp/TMscore", "pred.pdb", "native.pdb", "-seq"],
        )

    def test_cli_has_override_but_no_align_flag(self):
        parser = argparse.ArgumentParser()
        evaluation.add_arguments(parser)
        args = parser.parse_args(
            [
                "--pred-dir", "pred",
                "--native-dir", "native",
                "--tmscore", "custom",
            ]
        )
        self.assertEqual(args.tmscore, "custom")
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertNotIn("--align", option_strings)


if __name__ == "__main__":
    unittest.main()
