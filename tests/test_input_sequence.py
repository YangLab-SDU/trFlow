import tempfile
import unittest
from pathlib import Path

from trflow.cli import _load_predict_config, apply_cli, build_parser
from trflow.config import SampleConfig
from trflow.core import parse_a3m
from trflow.openfold_runner import _write_fasta
from trflow.validation import (
    read_a3m_query_sequence,
    resolve_target_sequence,
)


class TargetSequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write(self, name: str, content: str) -> Path:
        path = self.root / name
        path.write_text(content)
        return path

    def test_sequence_is_inferred_from_wrapped_a3m_query(self):
        msa = self._write(
            "target.a3m",
            ">query description\nACD\nEFG\n>hit\nACDefEFG\n",
        )
        header, sequence = read_a3m_query_sequence(str(msa))
        self.assertEqual(header, "query description")
        self.assertEqual(sequence, "ACDEFG")

        sample = SampleConfig(name="target", msa_path=str(msa))
        self.assertEqual(
            resolve_target_sequence(sample),
            ("ACDEFG", "msa_first_record", "query description"),
        )

        parsed = parse_a3m(str(msa))
        self.assertEqual(parsed.shape, (2, 6))

    def test_matching_fasta_is_accepted_and_normalized(self):
        msa = self._write("target.a3m", ">query\nACDEFG\n")
        fasta = self._write("target.fasta", ">target\nacdefg\n")
        sample = SampleConfig(
            name="target",
            msa_path=str(msa),
            fasta_path=str(fasta),
        )
        self.assertEqual(
            resolve_target_sequence(sample),
            ("ACDEFG", "fasta_validated_against_msa", "query"),
        )

    def test_openfold_fasta_is_generated_from_a3m_query(self):
        msa = self._write("target.a3m", ">query\nACDEFG\n")
        output = self.root / "openfold_input.fasta"
        _write_fasta(output, SampleConfig(name="target", msa_path=str(msa)), "target")
        self.assertEqual(output.read_text(), ">target\nACDEFG\n")

    def test_same_length_sequence_mismatch_is_rejected(self):
        msa = self._write("target.a3m", ">query\nACDEFG\n")
        fasta = self._write("target.fasta", ">target\nACNEFG\n")
        sample = SampleConfig(
            name="target",
            msa_path=str(msa),
            fasta_path=str(fasta),
        )
        with self.assertRaisesRegex(ValueError, "position 3"):
            resolve_target_sequence(sample)

    def test_a3m_query_markup_is_rejected(self):
        for sequence, message in (
            ("ACdEFG", "lowercase"),
            ("AC-EFG", "gap"),
            ("AC.EFG", "gap"),
        ):
            with self.subTest(sequence=sequence):
                msa = self._write("invalid.a3m", f">query\n{sequence}\n")
                with self.assertRaisesRegex(ValueError, message):
                    read_a3m_query_sequence(str(msa))

    def test_invalid_target_residue_is_rejected(self):
        msa = self._write("invalid.a3m", ">query\nAC:EFG\n")
        with self.assertRaisesRegex(ValueError, "invalid residue"):
            read_a3m_query_sequence(str(msa))


class DirectInputCliTests(unittest.TestCase):
    def _load(self, arguments):
        parser = build_parser()
        args = parser.parse_args(["predict", *arguments])
        return _load_predict_config(args, parser)

    def test_a3m_is_a_direct_input(self):
        cfg, mode, label = self._load(["target.a3m"])
        sample = cfg.samples[0]
        self.assertEqual(mode, "A3M")
        self.assertEqual(label, "target.a3m")
        self.assertEqual(sample.name, "target")
        self.assertEqual(sample.msa_path, "target.a3m")
        self.assertIsNone(sample.fasta_path)
        self.assertEqual(cfg.options.sample_num, 200)

    def test_a3m_accepts_optional_validation_fasta(self):
        cfg, mode, _label = self._load(
            ["target.a3m", "--fasta", "target.fasta"]
        )
        self.assertEqual(mode, "A3M + FASTA validation")
        self.assertEqual(cfg.samples[0].fasta_path, "target.fasta")

    def test_sample_num_can_override_default(self):
        parser = build_parser()
        args = parser.parse_args(
            ["predict", "target.a3m", "--sample-num", "7"]
        )
        cfg, _mode, _label = _load_predict_config(args, parser)
        self.assertEqual(apply_cli(cfg, args).options.sample_num, 7)

    def test_positional_fasta_mode_remains_supported(self):
        cfg, mode, _label = self._load(
            ["target.fasta", "--msa", "target.a3m"]
        )
        self.assertEqual(mode, "FASTA + A3M")
        self.assertEqual(cfg.samples[0].fasta_path, "target.fasta")

    def test_legacy_fasta_mode_remains_supported(self):
        cfg, mode, _label = self._load(
            ["--fasta", "target.fasta", "--msa", "target.a3m"]
        )
        self.assertEqual(mode, "FASTA + A3M")
        self.assertEqual(cfg.samples[0].msa_path, "target.a3m")


if __name__ == "__main__":
    unittest.main()
