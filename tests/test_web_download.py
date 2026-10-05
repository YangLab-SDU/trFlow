"""Portable aligned-result downloads and 2akl examples, using isolated data.

No prediction worker starts. The tiny synthetic ensemble is aligned by the
real CPU analysis subprocess; existing/user prediction outputs are never used.
"""
from __future__ import annotations

import csv
import hashlib
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
from http.server import ThreadingHTTPServer
from urllib.parse import quote, unquote
import zipfile

from trflow.web import JobManager, REPO, create_handler, read_json, write_json


ASSIGNMENT_FIELDS = [
    "index", "file", "cluster", "is_representative", "rmsd_to_reference", "mean_plddt", "model",
]
SUMMARY_FIELDS = [
    "cluster", "size", "representative_index", "representative_file",
    "representative_prediction_file", "mean_rmsd", "max_rmsd",
]


def snapshot(directory: Path) -> dict[str, bytes]:
    return {
        file.relative_to(directory).as_posix(): file.read_bytes()
        for file in directory.rglob("*") if file.is_file()
    }


def write_synthetic_ensemble(output: Path, name: str) -> None:
    predictions = output / "predictions"
    predictions.mkdir(parents=True)
    reference = [
        (0., 0., 0.), (3.8, .3, .5), (6.3, 2.9, .9),
        (5., 6.2, 2.), (1.3, 6.8, 3.2), (-1.2, 4., 4.),
    ]
    # Conformations 0 and 1 differ only by rigid motion. The third also bends.
    # Nonzero translations make the alignment-vs-original byte check decisive.
    rigid = [(-y + 20., x - 11., z + 7.) for x, y, z in reference]
    bent = [(-x - 14., -y + 9., z + 3.) for x, y, z in reference]
    bent[-1] = (bent[-1][0] + 3., bent[-1][1] + 1., bent[-1][2] - 2.)
    entries = []
    for index, positions in enumerate([reference, rigid, bent], start=1):
        filename = f"qa_download_{index:03}.pdb"
        atoms = []
        for residue, (x, y, z) in enumerate(positions, start=1):
            for atom, dx, dy, dz, element in (
                ("N", -.7, -.3, 0, "N"), ("CA", 0, 0, 0, "C"),
                ("C", .7, .3, 0, "C"), ("O", .9, 1.1, .2, "O"),
            ):
                atoms.append(
                    f"ATOM  {len(atoms) + 1:5d} {atom:>4s} ALA A{residue:4d}    "
                    f"{x + dx:8.3f}{y + dy:8.3f}{z + dz:8.3f}"
                    f"  1.00 80.00          {element:>2s}  \n"
                )
        (predictions / filename).write_text("".join(atoms) + "END\n", encoding="utf-8")
        entries.append({"file": filename, "index": index, "model": "NMR", "mean_plddt": .8})
    write_json(output / "info.json", {
        "sample_name": name, "qa_fixture": True, "sequence": {"length": 6},
        "predictions": entries, "options": {"sample_num": 3, "models": ["NMR"]},
        "stage_times": {"total": {"seconds": 3.25}},
    })


class DownloadArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trflow-download-qa-")
        self.root = Path(self.temporary.name)
        self.manager = JobManager(self.root / "web", import_existing=False, start_worker=False)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(self.manager))
        self.server.daemon_threads = True
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.manager.close()
        self.temporary.cleanup()

    def request(self, method: str, path: str, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            payload = json.dumps(body).encode("utf-8") if body is not None else None
            connection.request(method, path, payload, {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def fixture(self, imported=False, name="2akl"):
        if imported:
            repository = self.root / "external_repository"
            output = repository / "outputs" / "qa_imported"
            write_synthetic_ensemble(output, name)
            target_id = "existing-" + hashlib.sha256(str(output).encode()).hexdigest()[:16]
            with mock.patch("trflow.web.REPO", repository):
                self.manager._import_existing()
            record = self.manager.store.get(target_id)
        else:
            submitted = self.manager.submit([{
                "name": name, "msa_text": ">query\nACDEFG\n",
                "options": {"sample_num": 3, "models": ["NMR"], "geometric_exploration": False},
            }])[0]
            record = self.manager.store.get(submitted["id"])
            output = Path(record["output_path"])
            write_synthetic_ensemble(output, record["name"])
            record = self.manager.store.update(record["id"], status="completed", stage=5, generated=3)
        job = Path(record["job_path"])
        job.mkdir(parents=True, exist_ok=True)
        (job / "run.log").write_text("QA synthetic prediction log\n", encoding="utf-8")
        return record, output

    def download(self, record, k):
        status, headers, body = self.request("GET", f"/api/targets/{record['id']}/download?k={k}")
        self.assertEqual(status, 200, body[:400])
        self.assertEqual(headers["Content-Type"], "application/zip")
        self.assertEqual(headers["Content-Disposition"], f'attachment; filename="trflow_{record["name"]}.zip"')
        self.assertEqual(int(headers["Content-Length"]), len(body))
        archive = zipfile.ZipFile(io.BytesIO(body))
        self.assertIsNone(archive.testzip())
        return archive

    def test_archive_has_portable_layout_and_aligned_not_original_coordinates(self):
        record, output = self.fixture()
        original = snapshot(output)
        result = self.manager.analysis(record["id"], 2)
        with self.download(record, 2) as archive:
            names = archive.namelist()
            self.assertNotIn("clusters.json", names)
            self.assertFalse(any(name.startswith(("predictions/", "align/", "aligned/")) for name in names))
            self.assertEqual(len(names), len(set(names)), "ZIP entries must not be duplicated")
            prediction_paths = sorted(name for name in names if name.startswith("prediction/"))
            self.assertEqual(prediction_paths, sorted("prediction/" + item["filename"] for item in result["structures"]))
            different = 0
            for structure in result["structures"]:
                archived = archive.read("prediction/" + structure["filename"])
                aligned = Path(record["job_path"]) / "analysis" / structure["aligned_file"]
                self.assertEqual(archived, aligned.read_bytes())
                different += archived != original["predictions/" + structure["filename"]]
            self.assertGreaterEqual(different, 2)
            self.assertEqual(json.loads(archive.read("info.json")), read_json(output / "info.json"))
            self.assertIn(b"QA synthetic prediction log", archive.read("run.log"))
            readme = archive.read("README.txt").decode("utf-8")
            self.assertIn("zero-based", readme)
            self.assertIn("aligned", readme.lower())
            self.assertIn("Clusters: 2 (requested: 2)", readme)
            exported = json.loads(archive.read("clusters/clusters.json"))
            self.assertEqual(exported["reference"]["file"], "prediction/qa_download_001.pdb")
            self.assertEqual(exported["target"]["id"], record["id"])
            for structure in exported["structures"]:
                expected = "prediction/" + structure["filename"]
                self.assertEqual(structure["file"], expected)
                self.assertEqual(structure["aligned_file"], expected)
                self.assertNotIn("url", structure)
                self.assertNotIn("original_url", structure)
        self.assertEqual(snapshot(output), original)
        # Export normalization must not rewrite the API result or cached paths.
        unchanged = self.manager.analysis(record["id"], 2)
        self.assertEqual(unchanged["reference"]["file"], "qa_download_001.pdb")
        self.assertEqual([item["file"] for item in unchanged["structures"]], [item["filename"] for item in unchanged["structures"]])
        self.assertTrue(all(item["aligned_file"].startswith("aligned/") for item in unchanged["structures"]))
        self.assertTrue(all("url" in item and "original_url" in item for item in unchanged["structures"]))

    def test_requested_k_controls_members_representatives_and_statistics(self):
        record, output = self.fixture()
        original = snapshot(output)
        for requested, effective in ((1, 1), (2, 2), (3, 3), (99, 3)):
            with self.subTest(requested=requested), self.download(record, requested) as archive:
                result = json.loads(archive.read("clusters/clusters.json"))
                self.assertEqual((result["count"], result["k"], result["requested_k"]), (3, effective, requested))
                self.assertEqual(len(result["clusters"]), effective)
                representatives = [name for name in archive.namelist() if name.startswith("clusters/representatives/")]
                self.assertEqual(len(representatives), effective)
                self.assertEqual(sorted(index for cluster in result["clusters"] for index in cluster["members"]), [0, 1, 2])
                self.assertEqual(sum(item["representative"] for item in result["structures"]), effective)
                if effective == 2:
                    self.assertEqual([cluster["members"] for cluster in result["clusters"]], [[0, 1], [2]])
                for cluster in result["clusters"]:
                    members = cluster["members"]
                    representative = cluster["representative"]
                    self.assertEqual(cluster["count"], len(members))
                    self.assertEqual(cluster["size"], len(members))
                    self.assertIn(representative, members)
                    structure = result["structures"][representative]
                    expected_representative = f"clusters/representatives/cluster_{cluster['id']:03d}_{structure['filename']}"
                    self.assertEqual(cluster["representative_file"], expected_representative)
                    self.assertEqual(cluster["representative_prediction_file"], structure["file"])
                    self.assertEqual(cluster["member_files"], [result["structures"][index]["file"] for index in members])
                    self.assertEqual(archive.read(expected_representative), archive.read(structure["file"]))
                    self.assertEqual(archive.read(expected_representative),
                                     (Path(record["job_path"]) / "analysis" / "aligned" / structure["filename"]).read_bytes())
                    distances = result["distance_matrix"]
                    pairs = [distances[left][right] for position, left in enumerate(members) for right in members[position + 1:]]
                    self.assertAlmostEqual(cluster["mean_rmsd"], sum(pairs) / len(pairs) if pairs else 0., delta=2e-6)
                    self.assertAlmostEqual(cluster["max_rmsd"], max(pairs) if pairs else 0., delta=2e-6)
                    sums = [sum(distances[index][other] for other in members) for index in members]
                    self.assertAlmostEqual(sum(distances[representative][other] for other in members), min(sums), delta=2e-6)
                assignments_reader = csv.DictReader(io.StringIO(archive.read("clusters/assignments.csv").decode("utf-8")))
                self.assertEqual(assignments_reader.fieldnames, ASSIGNMENT_FIELDS)
                assignments = list(assignments_reader)
                self.assertEqual(len(assignments), 3)
                for row, structure in zip(assignments, result["structures"]):
                    self.assertEqual(int(row["index"]), structure["index"])
                    self.assertEqual(row["file"], structure["file"])
                    self.assertEqual(int(row["cluster"]), structure["cluster"])
                    self.assertEqual(int(row["is_representative"]), int(structure["representative"]))
                    self.assertEqual(float(row["rmsd_to_reference"]), structure["rmsd_to_reference"])
                    self.assertEqual(float(row["mean_plddt"]), structure["mean_plddt"])
                    self.assertEqual(row["model"], structure["model"])
                summary_reader = csv.DictReader(io.StringIO(archive.read("clusters/summary.csv").decode("utf-8")))
                self.assertEqual(summary_reader.fieldnames, SUMMARY_FIELDS)
                summary = list(summary_reader)
                self.assertEqual(len(summary), effective)
                for row, cluster in zip(summary, result["clusters"]):
                    self.assertEqual(int(row["cluster"]), cluster["id"])
                    self.assertEqual(int(row["size"]), cluster["count"])
                    self.assertEqual(int(row["representative_index"]), cluster["representative"])
                    self.assertEqual(row["representative_file"], cluster["representative_file"])
                    self.assertEqual(row["representative_prediction_file"], cluster["representative_prediction_file"])
                    self.assertEqual(float(row["mean_rmsd"]), cluster["mean_rmsd"])
                    self.assertEqual(float(row["max_rmsd"]), cluster["max_rmsd"])
        self.assertEqual(snapshot(output), original)

    def test_stale_aligned_cache_files_are_not_globbed_into_archive(self):
        record, output = self.fixture()
        self.manager.analysis(record["id"], 2)
        stale = Path(record["job_path"]) / "analysis" / "aligned" / "stale_other_target.pdb"
        stale.write_text("QA unrelated stale cache\n", encoding="utf-8")
        original = snapshot(output)
        with self.download(record, 2) as archive:
            self.assertFalse(any("stale_other_target" in name for name in archive.namelist()))
            self.assertEqual(len([name for name in archive.namelist() if name.startswith("prediction/")]), 3)
        self.assertEqual(stale.read_text(), "QA unrelated stale cache\n")
        self.assertEqual(snapshot(output), original)

    def test_download_uses_target_name_not_internal_identifier_and_preserves_imported_outputs(self):
        for imported in (False, True):
            with self.subTest(imported=imported):
                record, output = self.fixture(imported=imported)
                original = snapshot(output)
                self.assertEqual(record["existing"], imported)
                self.assertEqual(record["name"], "2akl")
                self.assertNotEqual(record["id"], "2akl")
                with self.download(record, 2) as archive:
                    self.assertEqual(json.loads(archive.read("clusters/clusters.json"))["target"]["id"], record["id"])
                self.assertEqual(snapshot(output), original)
                if imported:
                    self.assertTrue(record["id"].startswith("existing-"))
                    self.assertFalse(output.is_relative_to(self.manager.data_dir))
                    self.assertFalse((output / "analysis").exists())

    def test_download_names_are_safe_and_unicode_uses_utf8_content_disposition(self):
        record, output = self.fixture()
        original = snapshot(output)
        cases = (
            ("2akl", "different_sample_name", "trflow_2akl.zip"),
            (' ../2akl\\:"*?<>|\r\n injected ', "unused", "trflow_2akl_injected.zip"),
            ('..\\ / :*?<>|\r\n"', "unused", "trflow_target.zip"),
            ("蛋白 2α/测试", "unused", "trflow_蛋白_2α_测试.zip"),
            ("X" * 140, "unused", "trflow_" + "X" * 120 + ".zip"),
            ("", "fallback_sample", "trflow_fallback_sample.zip"),
            ("", "", "trflow_target.zip"),
        )
        archive_bytes = b"QA filename-only archive stub"
        with mock.patch.object(self.manager, "archive", return_value=archive_bytes) as archive:
            for name, sample_name, expected in cases:
                with self.subTest(name=name, sample_name=sample_name):
                    self.manager.store.update(record["id"], name=name, sample_name=sample_name)
                    status, headers, payload = self.request("GET", f"/api/targets/{record['id']}/download?k=2")
                    self.assertEqual(status, 200, payload)
                    self.assertEqual(payload, archive_bytes)
                    disposition = headers["Content-Disposition"]
                    self.assertTrue(disposition.isascii())
                    self.assertNotIn("\r", disposition)
                    self.assertNotIn("\n", disposition)
                    if expected.isascii():
                        self.assertEqual(disposition, f'attachment; filename="{expected}"')
                    else:
                        self.assertEqual(disposition, 'attachment; filename="trflow_target.zip"; '
                                         + "filename*=UTF-8''" + quote(expected, safe=""))
                        self.assertEqual(unquote(disposition.split("filename*=UTF-8''", 1)[1]), expected)
                    self.assertNotIn(record["id"], disposition)
                    archive.assert_called_with(record["id"], 2)
        self.assertEqual(snapshot(output), original)
        self.assertFalse((Path(record["job_path"]) / "analysis").exists())

    def test_invalid_cluster_counts_return_400_without_work_or_output_changes(self):
        record, output = self.fixture()
        original = snapshot(output)
        with mock.patch("trflow.web.subprocess.Popen", side_effect=AssertionError("Invalid k must not start analysis")):
            for value in ("0", "-1", "nope", "1.5", "true"):
                with self.subTest(k=value):
                    status, headers, payload = self.request("GET", f"/api/targets/{record['id']}/download?k={value}")
                    self.assertEqual(status, 400, payload)
                    self.assertNotIn("Content-Disposition", headers)
                    self.assertIn("error", json.loads(payload))
        self.assertEqual(snapshot(output), original)
        self.assertFalse((Path(record["job_path"]) / "analysis").exists())

    def test_unfinished_and_unknown_targets_cannot_download(self):
        record, output = self.fixture()
        self.manager.store.update(record["id"], status="queued")
        original = snapshot(output)
        with mock.patch("trflow.web.subprocess.Popen", side_effect=AssertionError("An unfinished target must not start analysis")):
            self.assertEqual(self.request("GET", f"/api/targets/{record['id']}/download?k=2")[0], 400)
            self.assertEqual(self.request("GET", "/api/targets/not-a-target/download?k=2")[0], 404)
        self.assertEqual(snapshot(output), original)

    def test_get_example_returns_exact_2akl_msa_without_adding_targets(self):
        source = REPO / "example" / "msa" / "2akl.a3m"
        original = source.read_bytes()
        status, _, payload = self.request("GET", "/api/example")
        self.assertEqual(status, 200, payload)
        example = json.loads(payload)
        self.assertEqual(example["name"], "2akl")
        self.assertEqual(example["msa_filename"], "2akl.a3m")
        self.assertEqual(example["msa_text"], source.read_text(encoding="utf-8"))
        self.assertEqual(self.manager.store.list(), [])
        self.assertEqual(list(self.manager.jobs_dir.iterdir()), [])
        self.assertEqual(source.read_bytes(), original)

    def test_post_example_uses_same_2akl_source_with_worker_disabled(self):
        source = REPO / "example" / "msa" / "2akl.a3m"
        original = source.read_bytes()
        with mock.patch("trflow.web.subprocess.Popen", side_effect=AssertionError("QA example must not start a prediction worker")):
            status, _, payload = self.request("POST", "/api/example", {"options": {"sample_num": 3, "models": ["NMR"]}})
        self.assertEqual(status, 201, payload)
        submitted = json.loads(payload)["targets"]
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0]["name"], "2akl")
        self.assertEqual(submitted[0]["status"], "queued")
        record = self.manager.store.get(submitted[0]["id"])
        configuration = read_json(Path(record["job_path"]) / "request.json")
        self.assertEqual(Path(configuration["samples"][0]["msa_path"]).read_text(encoding="utf-8"), source.read_text(encoding="utf-8"))
        self.assertIsNone(self.manager.thread)
        self.assertIsNone(self.manager.process)
        self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
