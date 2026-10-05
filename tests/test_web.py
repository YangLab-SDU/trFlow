"""Exercise the portal's queue, persistence, and HTTP boundaries without a GPU."""
from __future__ import annotations

import http.client
import json
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from trflow.web import JobManager, contained, create_handler, read_json, write_json


def target(name="target", **changes):
    return {"name": name, "msa_text": ">query\nACDEFG\n>hit\nACDEFG\n", **changes}


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trflow-web-tests-")
        self.root = Path(self.temporary.name)
        self.manager = JobManager(self.root, import_existing=False, start_worker=False)

    def tearDown(self):
        self.manager.close()
        self.temporary.cleanup()

    def test_invalid_batch_writes_and_enqueues_nothing(self):
        with self.assertRaises(ValueError):
            self.manager.submit([target("valid"), target("invalid", msa_text=">query\nAC-DEFG\n")])
        self.assertEqual(self.manager.store.list(), [])
        self.assertEqual(list(self.manager.jobs_dir.iterdir()), [])

    def test_json_import_is_atomic_and_preserves_default_random_seed(self):
        config = {
            "output_dir": "../not-used-by-web",
            "samples": [
                {"name": "valid", "msa_path": "folder/one.a3m"},
                {"name": "invalid", "msa_path": "two.a3m"},
            ],
        }
        files = [
            {"name": "one.a3m", "content": ">query\nACDEFG\n"},
            {"name": "two.a3m", "content": ">query\nACDE-FG\n"},
        ]
        with self.assertRaises(ValueError):
            self.manager.import_config({"config": config, "files": files})
        self.assertEqual(self.manager.store.list(), [])
        self.assertEqual(list(self.manager.jobs_dir.iterdir()), [])
        config["samples"].pop()
        result = self.manager.import_config({"config": json.dumps(config), "files": files})[0]
        self.assertIsNone(result["options"]["seed"])
        self.assertEqual(result["sample_num"], 200)
        self.assertEqual(result["queue_position"], 1)
        record = self.manager.store.get(result["id"])
        request = read_json(Path(record["job_path"]) / "request.json")
        # Windows runners can expose TEMP through an 8.3 alias or a junction.
        # Compare physical paths, matching the manager's resolved data root.
        self.assertTrue(Path(request["output_dir"]).resolve().is_relative_to(self.root.resolve()))

    def test_non_object_options_and_boolean_integer_options_rejected(self):
        for options in ([], False, 0, "", {"sample_num": True}, {"seed": True}, {"models": ["unknown"]}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.manager.submit([target(options=options)])
        self.assertEqual(self.manager.store.list(), [])

    def test_queue_is_serial_in_submission_order_and_skips_cancelled(self):
        records = self.manager.submit([
            target("first", options={"sample_num": 2}),
            target("cancelled", options={"sample_num": 2}),
            target("third", options={"sample_num": 2}),
        ])
        self.manager.cancel(records[1]["id"])
        state = {"active": 0, "maximum": 0, "started": [], "analysis": []}
        guard = threading.Lock()
        manager = self.manager

        class FakePrediction:
            """Stand in for the GPU process, retaining the real _run workflow."""
            def __init__(self, command, **kwargs):
                config = read_json(Path(command[command.index("predict") + 1]))
                sample = config["samples"][0]
                self.name = sample["name"]
                self.polls = 0
                self.waited = False
                with guard:
                    state["active"] += 1
                    state["maximum"] = max(state["maximum"], state["active"])
                    state["started"].append(self.name)
                directory = Path(config["output_dir"]) / self.name
                directory.mkdir(parents=True)
                write_json(directory / "info.json", {
                    "predictions": [{"file": f"sample_{i:03}.pdb"} for i in range(config["options"]["sample_num"])],
                    "stage_times": {"total": {"seconds": 0.01}},
                })
                kwargs["stdout"].write("[2/3] Geometric exploration\n")
                kwargs["stdout"].flush()

            def poll(self):
                self.polls += 1
                return None if self.polls == 1 else 0

            def wait(self):
                if not self.waited:
                    with guard:
                        state["active"] -= 1
                    self.waited = True
                return 0

        def analyze(target_id, k, allow_active=False):
            self.assertEqual(state["active"], 0)
            self.assertTrue(allow_active)
            record = manager.store.get(target_id)
            self.assertEqual(record["status"], "analyzing")
            state["analysis"].append(record["sample_name"])
            return {}

        with mock.patch("trflow.web.subprocess.Popen", FakePrediction), mock.patch.object(manager, "analysis", side_effect=analyze):
            manager.thread = threading.Thread(target=manager._worker, daemon=True)
            manager.thread.start()
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if all(manager.store.get(record["id"])["status"] not in {"queued", "running", "analyzing"} for record in records):
                    break
                time.sleep(0.02)
            self.assertEqual([manager.store.get(record["id"])["status"] for record in records], ["completed", "cancelled", "completed"])
            manager.close()
        self.assertEqual(state["maximum"], 1)
        self.assertEqual(state["started"], ["first", "third"])
        self.assertEqual(state["analysis"], ["first", "third"])

    def test_restart_preserves_queued_and_completed_marks_interrupted_failed(self):
        records = self.manager.submit([target("queued"), target("completed"), target("interrupted")])
        self.manager.store.update(records[1]["id"], status="completed", stage=5)
        self.manager.store.update(records[2]["id"], status="running", stage=2)
        self.manager.close()
        self.manager = JobManager(self.root, import_existing=False, start_worker=False)
        restored = [self.manager.store.get(record["id"]) for record in records]
        self.assertEqual([record["status"] for record in restored], ["queued", "completed", "failed"])
        self.assertIn("中断", restored[-1]["error"])
        self.assertIsNotNone(restored[-1]["finished_at"])

    def test_data_directory_has_only_one_service(self):
        with self.assertRaises(RuntimeError):
            JobManager(self.root, import_existing=False, start_worker=False)

    def test_retry_reuses_finished_predictions_without_starting_gpu_again(self):
        result = self.manager.submit([target("recover", options={"sample_num": 2})])[0]
        record = self.manager.store.get(result["id"])
        output = Path(record["output_path"])
        (output / "predictions").mkdir(parents=True)
        for index in (1, 2):
            (output / "predictions" / f"sample_{index:03}.pdb").write_text("END\n")
        write_json(output / "info.json", {
            "predictions": [{"file": f"sample_{index:03}.pdb"} for index in (1, 2)],
            "stage_times": {"total": {"seconds": 9.0}},
        })
        self.manager.store.update(result["id"], status="failed", error="Analysis interrupted")
        retried = self.manager.retry(result["id"])
        self.assertEqual(retried[0]["id"], result["id"])
        self.assertEqual(len(self.manager.store.list()), 1)
        with mock.patch("trflow.web.subprocess.Popen", side_effect=AssertionError("GPU process must not restart")), mock.patch.object(self.manager, "analysis", return_value={}) as analyze:
            self.manager._run(self.manager.store.get(result["id"]))
        analyze.assert_called_once_with(result["id"], 10, allow_active=True)
        recovered = self.manager.store.get(result["id"])
        self.assertEqual(recovered["status"], "completed")
        self.assertEqual(recovered["generated"], 2)
        self.assertEqual(recovered["elapsed_seconds"], 9.0)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trflow-http-tests-")
        self.manager = JobManager(Path(self.temporary.name), import_existing=False, start_worker=False)
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

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            data = json.dumps(body).encode() if body is not None else None
            request_headers = {"Content-Type": "application/json", **(headers or {})}
            connection.request(method, path, body=data, headers=request_headers)
            response = connection.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            connection.close()

    def test_post_batch_and_detail_do_not_expose_local_output_paths(self):
        status, data, headers = self.request("POST", "/api/targets", {"targets": [target()]})
        self.assertEqual(status, 201)
        record = json.loads(data)["targets"][0]
        self.assertNotIn("job_path", record)
        self.assertNotIn("output_path", record)
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        status, data, _ = self.request("GET", "/api/targets/" + record["id"])
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["queue_position"], 1)

    def test_cross_site_submissions_and_foreign_hosts_are_rejected(self):
        status, _, _ = self.request("POST", "/api/targets", {"targets": [target()]}, {"Origin": "https://unrelated.example"})
        self.assertEqual(status, 400)
        status, _, _ = self.request("GET", "/api/config", headers={"Host": "unrelated.example"})
        self.assertEqual(status, 400)
        self.assertEqual(self.manager.store.list(), [])

    def test_detail_uses_log_tail_but_log_download_retains_full_history(self):
        record_id = self.manager.submit([target()])[0]["id"]
        record = self.manager.store.get(record_id)
        (Path(record["job_path"]) / "run.log").write_text(
            "\x1b[31mFIRST-LINE\x1b[0m\n" + "\n".join(f"line-{index}" for index in range(200)) + "\nLAST-LINE\n",
            encoding="utf-8",
        )
        status, detail, _ = self.request("GET", "/api/targets/" + record_id)
        self.assertEqual(status, 200)
        self.assertNotIn("FIRST-LINE", json.loads(detail)["log_tail"])
        self.assertIn("LAST-LINE", json.loads(detail)["log_tail"])
        status, complete, headers = self.request("GET", "/api/targets/" + record_id + "/log")
        self.assertEqual(status, 200)
        self.assertIn("FIRST-LINE", complete.decode())
        self.assertIn("LAST-LINE", complete.decode())
        self.assertNotIn(b"\x1b[", complete)
        self.assertTrue(headers["Content-Type"].startswith("text/plain"))

    def test_traversal_and_malformed_target_routes_are_rejected(self):
        for path in ("/static/%2e%2e/web.py", "/static/%2e%2e%5cweb.py", "/api/targets/", "/api/targets/nonexistent/structures/%2e%2e%5cprivate.pdb"):
            with self.subTest(path=path):
                status, _, _ = self.request("GET", path)
                self.assertIn(status, {400, 404})
        with self.assertRaises(ValueError):
            contained(self.manager.data_dir, "../private.pdb")


if __name__ == "__main__":
    unittest.main()
