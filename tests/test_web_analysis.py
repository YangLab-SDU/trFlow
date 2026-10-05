from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from trflow.web_analysis import _kabsch, _prediction_files, analyze_ensemble, cluster_ensemble


def write_pdb(path: Path, positions: np.ndarray) -> None:
    path.write_text("".join(
        f"ATOM  {index:5d}  CA  ALA A{index:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C  \n"
        for index, (x, y, z) in enumerate(positions, start=1)
    ) + "END\n", encoding="utf-8")


class EnsembleAnalysisTests(unittest.TestCase):
    def test_analysis_cli_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions = root / "predictions"
            predictions.mkdir()
            write_pdb(predictions / "one.pdb", np.eye(3))
            analysis = root / "analysis"
            result_path = root / "result.json"
            command = [sys.executable, "-B", "-m", "trflow.web_analysis",
                       "--predictions", str(predictions), "--analysis", str(analysis),
                       "--k", "3", "--result", str(result_path)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30,
                                       cwd=Path(__file__).resolve().parents[1])
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            import json
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual((result["count"], result["k"], result["requested_k"]), (1, 1, 3))
            # The worker also supports a cached re-cut without prediction inputs.
            recut = subprocess.run([sys.executable, "-B", "-m", "trflow.web_analysis",
                                   "--analysis", str(analysis), "--k", "1", "--result", str(result_path)],
                                  capture_output=True, text=True, timeout=30,
                                  cwd=Path(__file__).resolve().parents[1])
            self.assertEqual(recut.returncode, 0, recut.stdout + recut.stderr)
            self.assertEqual(json.loads(result_path.read_text())["requested_k"], 1)

    @unittest.skipUnless(os.name == "nt", "Conda DLL bootstrap is Windows-specific")
    def test_direct_interpreter_thread_without_conda_path(self):
        environment = os.environ.copy()
        library_bin = os.path.normcase(os.path.abspath(Path(sys.prefix) / "Library" / "bin"))
        environment["PATH"] = os.pathsep.join(
            value for value in environment.get("PATH", "").split(os.pathsep)
            if value and os.path.normcase(os.path.abspath(value)) != library_bin
        )
        script = (
            "import numpy as np; from concurrent.futures import ThreadPoolExecutor; "
            "from trflow.web_analysis import _kabsch; a=np.eye(3); "
            "r=ThreadPoolExecutor(1).submit(_kabsch,a,a).result(); "
            "assert r[2]<1e-10; print('threaded alignment passed')"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-X", "faulthandler", "-c", script],
            env=environment, capture_output=True, text=True, timeout=30,
            cwd=Path(__file__).resolve().parents[1],
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("threaded alignment passed", result.stdout)

    def test_prediction_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ["sample_1000.pdb", "sample_999.pdb", "sample_001.pdb"]:
                (root / name).touch()
            self.assertEqual(
                [file.name for file in _prediction_files(root, None)],
                ["sample_001.pdb", "sample_999.pdb", "sample_1000.pdb"],
            )
            metadata = {"predictions": [{"file": "sample_1000.pdb", "index": 1},
                                        {"file": "sample_999.pdb", "index": 2}]}
            self.assertEqual(
                [file.name for file in _prediction_files(root, metadata)],
                ["sample_1000.pdb", "sample_999.pdb", "sample_001.pdb"],
            )

    def test_rigid_motion_and_reflection(self):
        points = np.array([[0., 0., 0.], [3., 0., 0.], [0., 4., 0.], [0., 0., 5.]])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        transformed = points @ rotation + np.array([15., -9., 4.])
        fitted_rotation, translation, rmsd = _kabsch(transformed, points)
        self.assertLess(rmsd, 1e-10)
        np.testing.assert_allclose(transformed @ fitted_rotation + translation, points, atol=1e-10)
        reflected = points * np.array([-1., 1., 1.])
        self.assertGreater(_kabsch(reflected, points)[2], 0.5)

    def test_clustering_alignment_and_recut(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prediction_dir = root / "predictions"
            prediction_dir.mkdir()
            points = np.array([[0., 0., 0.], [3., 0., 0.], [0., 4., 0.], [0., 0., 5.]])
            bent = points.copy()
            bent[-1] += np.array([4., 2., 0.])
            rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
            for index, conformation in enumerate([points, points @ rotation + 10., bent, bent @ rotation - 7.]):
                write_pdb(prediction_dir / f"sample_{index:03d}.pdb", conformation)
            analysis_dir = root / "analysis"
            default = analyze_ensemble(prediction_dir, analysis_dir)
            self.assertEqual(default["k"], 4)
            result = cluster_ensemble(analysis_dir, 2)
            self.assertEqual([cluster["members"] for cluster in result["clusters"]], [[0, 1], [2, 3]])
            self.assertEqual(sum(item["representative"] for item in result["structures"]), 2)
            self.assertLess(result["distance_matrix"][0][1], 1e-6)
            self.assertLess(result["distance_matrix"][2][3], 1e-6)
            self.assertGreater(result["distance_matrix"][0][2], 0.5)
            self.assertEqual(result["length"], 4)
            self.assertEqual(len(result["embedding"]), 4)
            self.assertTrue(np.isfinite([[point["x"], point["y"]] for point in result["embedding"]]).all())
            self.assertTrue(all((analysis_dir / item["aligned_file"]).exists() for item in result["structures"]))
            self.assertEqual(result["structures"][0]["confidence"], 90.)
            cached_mtime = (analysis_dir / "analysis.json").stat().st_mtime_ns
            analyze_ensemble(prediction_dir, analysis_dir)
            self.assertEqual((analysis_dir / "analysis.json").stat().st_mtime_ns, cached_mtime)
            self.assertEqual(cluster_ensemble(analysis_dir, 1)["clusters"][0]["count"], 4)
            with self.assertRaises(ValueError):
                cluster_ensemble(analysis_dir, 0)

    def test_single_structure_and_empty_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                analyze_ensemble(root, root / "analysis")
            write_pdb(root / "single.pdb", np.array([[0., 0., 0.], [3., 0., 0.], [0., 4., 0.]]))
            result = analyze_ensemble(root, root / "analysis")
            self.assertEqual((result["n"], result["k"]), (1, 1))
            self.assertEqual(result["embedding"][0]["x"], 0.)
            self.assertEqual(result["clusters"][0]["representative"], 0)


class IsolatedAnalysisTests(unittest.TestCase):
    def test_native_ensemble_suite(self):
        """Numerical analysis has its own process, just like the web worker.

        Existing inference tests load Torch's Intel OpenMP runtime; this Conda
        NumPy build loads LLVM OpenMP. A clean child avoids mixing those DLLs.
        """
        completed = subprocess.run(
            [sys.executable, "-B", "-X", "faulthandler", str(Path(__file__).resolve()), "-v"],
            capture_output=True, text=True, timeout=60,
            cwd=Path(__file__).resolve().parents[1],
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Ran 6 tests", completed.stderr)


def load_tests(loader, tests, pattern):
    # unittest discovery runs the subprocess wrapper; the child runs the checks.
    checks = EnsembleAnalysisTests if __name__ == "__main__" else IsolatedAnalysisTests
    return loader.loadTestsFromTestCase(checks)


if __name__ == "__main__":
    unittest.main()
