"""Recoverable deletion checks on isolated data only; no prediction worker starts."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import sqlite3
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from trflow.web import JobManager, create_handler, read_json, write_json


def fixture_target(manager: JobManager, name: str, status: str = "completed") -> dict:
    """Create explicit QA fixtures, never inference outputs or GPU work."""
    submitted = manager.submit([{
        "name": name, "msa_text": ">QA fixture query\nACDEFG\n",
        "options": {"sample_num": 3, "models": ["NMR"], "geometric_exploration": False},
    }])[0]
    record = manager.store.get(submitted["id"])
    job = Path(record["job_path"])
    (job / "marker.txt").write_text("QA-owned target data", encoding="utf-8")
    if status == "completed":
        output = Path(record["output_path"])
        predictions = output / "predictions"
        predictions.mkdir(parents=True)
        entries = []
        for conformation in range(3):
            filename = f"qa_fixture_{conformation + 1:03}.pdb"
            # A deliberately synthetic short backbone, including O atoms so
            # 3Dmol can render a ribbon. These are UI fixtures, not predictions.
            atoms = []
            for residue in range(1, 7):
                x = residue * 2.0
                y = 3.0 * math.sin(residue * .8)
                z = 3.0 * math.cos(residue * .8) + conformation * .2 * math.sin(residue)
                for atom, dx, dy, dz, element in (
                    ("N", -.7, -.3, 0, "N"), ("CA", 0, 0, 0, "C"),
                    ("C", .7, .3, 0, "C"), ("O", .9, 1.1, .2, "O"),
                ):
                    atoms.append(
                        f"ATOM  {len(atoms) + 1:5d} {atom:>4s} ALA A{residue:4d}    "
                        f"{x + dx:8.3f}{y + dy:8.3f}{z + dz:8.3f}"
                        f"  1.00 80.00          {element:>2s}  \n"
                    )
            pdb = "".join(atoms) + "END\n"
            (predictions / filename).write_text(pdb, encoding="utf-8")
            entries.append({"file": filename, "index": conformation + 1, "mean_plddt": .8, "model": "NMR"})
        write_json(output / "info.json", {
            "sample_name": record["sample_name"], "qa_fixture": True,
            "sequence": {"length": 6}, "predictions": entries,
            "options": record["options"], "stage_times": {"total": {"seconds": .03}},
        })
    return manager.store.update(record["id"], status=status,
                                stage=5 if status == "completed" else 0,
                                generated=3 if status == "completed" else 0)


class RecoverableDeleteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trflow-delete-qa-")
        self.root = Path(self.temporary.name)
        self.data = self.root / "web"
        self.manager = JobManager(self.data, import_existing=False, start_worker=False)
        self._start_server()

    def _start_server(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(self.manager))
        self.server.daemon_threads = True
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def _stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.manager.close()

    def tearDown(self):
        self._stop_server()
        self.temporary.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            data = json.dumps(body).encode() if body is not None else None
            connection.request(method, path, data, {"Content-Type": "application/json", **(headers or {})})
            response = connection.getresponse()
            payload = response.read()
            return response.status, json.loads(payload) if payload else {}
        finally:
            connection.close()

    def test_inactive_target_moves_to_recycle_and_restores_without_loss(self):
        for original_status in ("completed", "failed", "cancelled"):
            with self.subTest(status=original_status):
                record = fixture_target(self.manager, original_status, original_status)
                job = Path(record["job_path"])
                status, result = self.request("DELETE", "/api/targets/" + record["id"])
                self.assertEqual(status, 200, result)
                self.assertTrue(result["deleted"])
                self.assertTrue(result["can_restore"])
                self.assertEqual(result["id"], record["id"])
                self.assertFalse(job.exists())
                recycled = self.data / "recycle_bin" / record["id"] / "job"
                self.assertEqual((recycled / "marker.txt").read_text(), "QA-owned target data")
                with self.assertRaises(KeyError):
                    self.manager.store.get(record["id"])
                status, listing = self.request("GET", "/api/recycle-bin")
                self.assertEqual(status, 200)
                self.assertIn(record["id"], [item["id"] for item in listing["targets"]])
                status, restored = self.request("POST", f"/api/recycle-bin/{record['id']}/restore", {})
                self.assertEqual(status, 200, restored)
                self.assertEqual(restored["target"]["status"], original_status)
                self.assertEqual((job / "marker.txt").read_text(), "QA-owned target data")
                self.assertNotIn(record["id"], [item["id"] for item in self.request("GET", "/api/recycle-bin")[1]["targets"]])

    def test_deleted_target_is_hidden_from_all_read_and_retry_routes(self):
        record = fixture_target(self.manager, "hidden")
        self.assertEqual(self.request("DELETE", "/api/targets/" + record["id"])[0], 200)
        with mock.patch("trflow.web.subprocess.Popen", side_effect=AssertionError("Deleted targets must not spawn a process")):
            for suffix in ("", "/analysis", "/log", "/download", "/structures/qa_fixture_001.pdb"):
                with self.subTest(suffix=suffix):
                    self.assertEqual(self.request("GET", "/api/targets/" + record["id"] + suffix)[0], 404)
            self.assertEqual(self.request("POST", "/api/targets/" + record["id"] + "/retry", {})[0], 404)
        self.assertEqual(self.request("GET", "/api/targets")[1]["targets"], [])

    def test_active_targets_are_not_moved_or_hidden(self):
        for active in ("queued", "running", "analyzing"):
            with self.subTest(status=active):
                record = fixture_target(self.manager, active, active)
                status, result = self.request("DELETE", "/api/targets/" + record["id"])
                self.assertEqual(status, 409)
                self.assertEqual(result["code"], "target_active")
                self.assertEqual(self.manager.store.get(record["id"])["status"], active)
                self.assertTrue(Path(record["job_path"]).exists())

    def test_cancelled_but_running_process_remains_busy(self):
        record = fixture_target(self.manager, "pending termination", "cancelled")
        self.manager.running_id = record["id"]
        self.manager.process = mock.Mock(poll=mock.Mock(return_value=None))
        try:
            status, result = self.request("DELETE", "/api/targets/" + record["id"])
            self.assertEqual(status, 409)
            self.assertEqual(result["code"], "target_busy")
            self.assertTrue(Path(record["job_path"]).exists())
        finally:
            self.manager.running_id = None
            self.manager.process = None

    def test_analysis_and_delete_do_not_race_directory_move(self):
        record = fixture_target(self.manager, "analysis lock")
        entered, release = threading.Event(), threading.Event()
        errors = []

        class AnalysisProcess:
            def __init__(self, command, **kwargs):
                self.destination = Path(command[command.index("--result") + 1])
                self.returncode = 0

            def communicate(self, timeout=None):
                entered.set()
                release.wait(5)
                write_json(self.destination, {"structures": [], "count": 3, "k": 3})
                return b"QA fake analysis process\n", None

        def analyze():
            try:
                self.manager.analysis(record["id"])
            except Exception as error:
                errors.append(error)

        with mock.patch("trflow.web.subprocess.Popen", AnalysisProcess):
            worker = threading.Thread(target=analyze)
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                status, result = self.request("DELETE", "/api/targets/" + record["id"])
                self.assertEqual(status, 409, result)
                self.assertEqual(result["code"], "target_busy")
                self.assertTrue(Path(record["job_path"]).exists())
            finally:
                release.set()
                worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
        self.assertEqual(self.request("DELETE", "/api/targets/" + record["id"])[0], 200)

    def test_filesystem_failure_does_not_tombstone_record(self):
        record = fixture_target(self.manager, "filesystem failure")
        job = Path(record["job_path"])
        rename = Path.rename

        def refuse_move(source, destination):
            if source == job:
                raise OSError("QA simulated file lock")
            return rename(source, destination)

        with mock.patch.object(Path, "rename", refuse_move):
            self.assertIn(self.request("DELETE", "/api/targets/" + record["id"])[0], {400, 409, 500})
        self.assertEqual(self.manager.store.get(record["id"])["status"], "completed")
        self.assertTrue((job / "marker.txt").exists())
        self.assertEqual(self.request("GET", "/api/recycle-bin")[1]["targets"], [])

    def test_database_failure_rolls_back_directory_move(self):
        record = fixture_target(self.manager, "database failure")
        with mock.patch.object(self.manager.store, "update", side_effect=sqlite3.OperationalError("QA database failure")):
            self.assertIn(self.request("DELETE", "/api/targets/" + record["id"])[0], {400, 409, 500})
        self.assertTrue((Path(record["job_path"]) / "marker.txt").exists())
        self.assertEqual(self.manager.store.get(record["id"])["status"], "completed")

    def test_restore_refuses_to_overwrite_a_new_directory(self):
        record = fixture_target(self.manager, "restore conflict")
        self.assertEqual(self.request("DELETE", "/api/targets/" + record["id"])[0], 200)
        job = Path(record["job_path"])
        job.mkdir()
        (job / "marker.txt").write_text("New owner data")
        status, result = self.request("POST", f"/api/recycle-bin/{record['id']}/restore", {})
        self.assertEqual(status, 409)
        self.assertEqual(result["code"], "restore_conflict")
        self.assertEqual((job / "marker.txt").read_text(), "New owner data")
        recycled = self.data / "recycle_bin" / record["id"] / "job" / "marker.txt"
        self.assertEqual(recycled.read_text(), "QA-owned target data")

    def test_restore_database_failure_returns_job_to_recycle_without_unhiding(self):
        record = fixture_target(self.manager, "restore database failure")
        self.assertEqual(self.request("DELETE", "/api/targets/" + record["id"])[0], 200)
        entry = self.data / "recycle_bin" / record["id"]
        manifest_before = (entry / "manifest.json").read_bytes()
        with mock.patch.object(self.manager.store, "update", side_effect=sqlite3.OperationalError("QA restore database failure")):
            status, result = self.request("POST", f"/api/recycle-bin/{record['id']}/restore", {})
            self.assertIn(status, {400, 409, 500}, result)
        self.assertFalse(Path(record["job_path"]).exists())
        self.assertEqual((entry / "job" / "marker.txt").read_text(), "QA-owned target data")
        self.assertEqual((entry / "manifest.json").read_bytes(), manifest_before)
        self.assertEqual(list(self.manager.recycle_dir.glob(record["id"] + ".restored-*")), [])
        self.assertTrue(self.manager.store.get(record["id"], include_deleted=True)["deleted_at"])
        self.assertEqual(self.request("GET", "/api/targets/" + record["id"])[0], 404)
        self.assertEqual(self.request("POST", f"/api/recycle-bin/{record['id']}/restore", {})[0], 200)

    def test_corrupted_external_job_and_output_paths_are_never_moved(self):
        with tempfile.TemporaryDirectory(prefix="trflow-external-qa-") as external_directory:
            external = Path(external_directory)
            marker = external / "external-owner.txt"
            marker.write_text("External data must survive", encoding="utf-8")
            for field in ("job_path", "output_path"):
                with self.subTest(field=field):
                    record = fixture_target(self.manager, "corrupted " + field)
                    original_job = Path(record["job_path"])
                    self.manager.store.update(record["id"], **{field: str(external)})
                    status, result = self.request("DELETE", "/api/targets/" + record["id"])
                    self.assertEqual(status, 400, result)
                    self.assertEqual(marker.read_text(), "External data must survive")
                    self.assertEqual(sorted(path.name for path in external.iterdir()), ["external-owner.txt"])
                    self.assertTrue((original_job / "marker.txt").exists())
                    self.assertFalse(self.manager.store.get(record["id"]).get("deleted_at"))
                    self.assertNotIn(record["id"], [item["id"] for item in self.request("GET", "/api/recycle-bin")[1]["targets"]])

    def test_same_target_can_be_deleted_and_restored_twice(self):
        record = fixture_target(self.manager, "repeat deletion")
        for cycle in range(2):
            with self.subTest(cycle=cycle):
                self.assertEqual(self.request("DELETE", "/api/targets/" + record["id"])[0], 200)
                self.assertEqual(self.request("GET", "/api/targets/" + record["id"])[0], 404)
                status, restored = self.request("POST", f"/api/recycle-bin/{record['id']}/restore", {})
                self.assertEqual(status, 200, restored)
                self.assertEqual(restored["target"]["id"], record["id"])
                self.assertEqual(restored["target"]["status"], "completed")
                self.assertEqual((Path(record["job_path"]) / "marker.txt").read_text(), "QA-owned target data")
                self.assertEqual(self.request("GET", "/api/recycle-bin")[1]["targets"], [])

    def test_cancelled_target_finish_analysis_never_resumes_or_spawns(self):
        stale_record = fixture_target(self.manager, "cancelled after prediction")
        self.manager.store.update(stale_record["id"], status="cancelled")
        with mock.patch.object(self.manager, "analysis", side_effect=AssertionError("Cancelled target must not be analyzed")) as analysis:
            with mock.patch("trflow.web.subprocess.Popen", side_effect=AssertionError("Cancelled target must not spawn")) as spawn:
                self.manager._finish_analysis(stale_record)
        analysis.assert_not_called()
        spawn.assert_not_called()
        self.assertEqual(self.manager.store.get(stale_record["id"])["status"], "cancelled")

    def test_cancel_during_finish_analysis_is_not_overwritten_by_completed(self):
        record = fixture_target(self.manager, "cancel during analysis")
        cancelled = {}

        def cancel_while_analyzing(target_id, *args, **kwargs):
            self.assertEqual(self.manager.store.get(target_id)["status"], "analyzing")
            cancelled.update(self.manager.cancel(target_id))
            return {"structures": [], "count": 3, "k": 3}

        with mock.patch.object(self.manager, "analysis", side_effect=cancel_while_analyzing):
            self.manager._finish_analysis(record)
        final = self.manager.store.get(record["id"])
        self.assertEqual(final["status"], "cancelled")
        self.assertEqual(final["stage"], 4)
        self.assertEqual(final["finished_at"], cancelled["finished_at"])

    def test_imported_outputs_survive_delete_and_do_not_reappear_after_restart(self):
        repository = self.root / "fake_repository"
        source = repository / "outputs" / "imported"
        (source / "predictions").mkdir(parents=True)
        original = source / "predictions" / "original.pdb"
        original.write_text("Original external QA PDB\n")
        write_json(source / "info.json", {"sample_name": "imported", "predictions": [{"file": "original.pdb"}]})
        target_id = "existing-" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
        with mock.patch("trflow.web.REPO", repository):
            self.manager._import_existing()
        record = self.manager.store.get(target_id)
        cache = Path(record["job_path"]) / "analysis"
        cache.mkdir(parents=True)
        (cache / "cache-marker.txt").write_text("Web cache")
        self.assertEqual(self.request("DELETE", "/api/targets/" + target_id)[0], 200)
        self.assertEqual(original.read_text(), "Original external QA PDB\n")
        self._stop_server()
        with mock.patch("trflow.web.REPO", repository):
            self.manager = JobManager(self.data, import_existing=True, start_worker=False)
        self._start_server()
        self.assertEqual(self.request("GET", "/api/targets")[1]["targets"], [])
        self.assertEqual(self.request("GET", "/api/targets/" + target_id)[0], 404)
        status, restored = self.request("POST", f"/api/recycle-bin/{target_id}/restore", {})
        self.assertEqual(status, 200, restored)
        self.assertTrue(restored["target"]["existing"])
        self.assertEqual(original.read_text(), "Original external QA PDB\n")
        self.assertEqual((cache / "cache-marker.txt").read_text(), "Web cache")

    def test_cross_site_delete_is_rejected_and_unknown_ids_are_not_found(self):
        record = fixture_target(self.manager, "origin check")
        status, _ = self.request("DELETE", "/api/targets/" + record["id"], headers={"Origin": "https://unrelated.example"})
        self.assertEqual(status, 400)
        self.assertTrue(Path(record["job_path"]).exists())
        self.assertEqual(self.request("DELETE", "/api/targets/not-a-target")[0], 404)


def serve_fixture(args):
    """Optional isolated UI fixture service. Contains no actual predictions."""
    manager = JobManager(Path(args.data_dir), import_existing=False, start_worker=False)
    if not manager.store.list():
        fixture_target(manager, "QA completed fixture")
        fixture_target(manager, "QA failed fixture", "failed")
        fixture_target(manager, "QA active fixture", "queued")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), create_handler(manager))
    server.daemon_threads = True
    print(f"Isolated QA fixture at http://127.0.0.1:{args.port}; GPU worker disabled", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.close()


if __name__ == "__main__":
    if "--serve-fixture" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--serve-fixture", action="store_true")
        parser.add_argument("--port", type=int, default=8766)
        parser.add_argument("--data-dir", required=True)
        serve_fixture(parser.parse_args())
    else:
        unittest.main()
