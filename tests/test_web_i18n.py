"""Controlled web errors are localizable; model diagnostics remain untouched.

All records and HTTP requests use a disposable server with its worker disabled.
No existing prediction, GPU process, or user's web-service data is accessed.
"""
from __future__ import annotations

import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest import mock

from trflow import web_errors
from trflow.web import JobManager, STATIC, create_handler, read_json, write_json
from trflow.web_errors import error_fields


INTERRUPTED = "服务已中断；可重新提交此目标。"
RAW_DIAGNOSTIC = "RuntimeError: 模型原始诊断 — tensor shape mismatch\n  at predictor.py:42"


class WebErrorI18nTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trflow-i18n-qa-")
        self.directory = Path(self.temporary.name)
        self.manager = JobManager(self.directory, import_existing=False, start_worker=False)
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

    def target(self, name="QA controlled error", **changes):
        public = self.manager.submit([{
            "name": name, "msa_text": ">query\nACDEFG\n",
            "options": {"sample_num": 3, "models": ["NMR"], "geometric_exploration": False},
        }])[0]
        return self.manager.store.update(public["id"], **changes)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            encoded = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8") if body is not None else None
            connection.request(method, path, body=encoded, headers={"Content-Type": "application/json", **(headers or {})})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_legacy_service_interrupted_is_localizable_without_rewriting_history(self):
        record = self.target(status="failed", error=INTERRUPTED)
        original = self.manager.store.get(record["id"])
        (Path(record["job_path"]) / "run.log").write_text(RAW_DIAGNOSTIC + "\n", encoding="utf-8")
        for path in ("/api/targets", "/api/targets/" + record["id"]):
            with self.subTest(path=path):
                status, result = self.request("GET", path)
                self.assertEqual(status, 200)
                target = result["targets"][0] if "targets" in result else result
                self.assertEqual(target["error"], INTERRUPTED)
                self.assertEqual(target["error_code"], "service_interrupted")
                self.assertEqual(target["error_params"], {})
                if "log_tail" in target:
                    self.assertEqual(target["log_tail"], RAW_DIAGNOSTIC)
        self.assertEqual(self.manager.store.get(record["id"]), original)

    def test_structured_failure_parameters_and_raw_error_are_preserved(self):
        record = self.target(status="failed", error="仅生成 2/3 个构象",
                             error_code="incomplete_predictions", error_params={"generated": 2, "expected": 3})
        original = self.manager.store.get(record["id"])
        status, result = self.request("GET", "/api/targets/" + record["id"])
        self.assertEqual(status, 200)
        self.assertEqual(result["error"], "仅生成 2/3 个构象")
        self.assertEqual(result["error_code"], "incomplete_predictions")
        self.assertEqual(result["error_params"], {"generated": 2, "expected": 3})
        self.assertEqual(self.manager.store.get(record["id"]), original)

    def test_unknown_model_diagnostic_is_not_misclassified_as_a_controlled_error(self):
        # Even a known phrase inside a trace must not translate the whole trace.
        diagnostics = [RAW_DIAGNOSTIC, "RuntimeError: " + INTERRUPTED + "\n  at predictor.py:42"]
        for prefix in ("RuntimeError:", "ValueError:", "torch.linalg.LinAlgError:", "Traceback (most recent call last):"):
            diagnostics.extend((prefix + " toy 缺少 A3M 内容", prefix + " toy 缺少 A3M 内容\n  at predictor.py:42"))
        for raw in diagnostics:
            with self.subTest(raw=raw):
                record = self.target(status="failed", error=raw)
                original = self.manager.store.get(record["id"])
                public = self.manager.public(record)
                self.assertEqual(public["error"], raw)
                self.assertIsNone(public.get("error_code"))
                self.assertIn(public.get("error_params"), (None, {}))
                self.assertEqual(self.manager.store.get(record["id"]), original)
        self.assertEqual(error_fields(ValueError("toy 缺少 A3M 内容")), {
            "error": "toy 缺少 A3M 内容", "error_code": "msa_missing", "error_params": {"name": "toy"},
        })
        explicit = self.target(status="failed", error=RAW_DIAGNOSTIC,
                               error_code="analysis_failed", error_params={"detail": RAW_DIAGNOSTIC})
        public = self.manager.public(explicit)
        self.assertEqual(public["error"], RAW_DIAGNOSTIC)
        self.assertEqual(public["error_code"], "analysis_failed")
        self.assertEqual(public["error_params"], {"detail": RAW_DIAGNOSTIC})

    def test_restart_persists_structured_interruption_without_starting_a_worker(self):
        record = self.target(status="running", stage=2)
        self.manager.close()
        with mock.patch("trflow.web.subprocess.Popen", side_effect=AssertionError("I18n QA cannot start prediction")):
            self.manager = JobManager(self.directory, import_existing=False, start_worker=False)
        restored = self.manager.store.get(record["id"])
        self.assertEqual(restored["status"], "failed")
        self.assertEqual(restored["error_code"], "service_interrupted")
        self.assertEqual(restored["error_params"], {})
        self.assertEqual(restored["error"], INTERRUPTED)
        self.assertIsNotNone(restored["finished_at"])
        self.assertIsNone(self.manager.thread)
        self.assertIsNone(self.manager.process)

    def test_http_input_errors_have_parameters_and_never_queue_invalid_targets(self):
        valid = {"name": "QA 用户原文", "msa_text": ">query\nACDEFG\n"}

        def with_option(options):
            return {"targets": [{**valid, "options": options}]}
        cases = (
            ("/api/targets", b"{invalid-json", None, "invalid_json", None),
            ("/api/targets", None, None, "invalid_body_size", {}),
            ("/api/targets", {}, {"Content-Length": str(64 * 1024 * 1024 + 1)}, "invalid_body_size", {}),
            ("/api/targets", {}, {"Content-Type": "text/plain"}, "json_content_required", {}),
            ("/api/targets", [], None, "json_object_required", {}),
            ("/api/targets", {"targets": []}, None, "invalid_target_count", {}),
            ("/api/targets", {"targets": [{}]}, None, "invalid_target_name", {}),
            ("/api/targets", with_option([]), None, "options_object", {}),
            ("/api/targets", with_option({"unknown": 1}), None, "unknown_options", {"fields": "unknown"}),
            ("/api/targets", with_option({"sample_num": 0}), None, "invalid_positive_integer", {"field": "sample_num"}),
            ("/api/targets", with_option({"steps": 101}), None, "invalid_positive_integer", {"field": "steps"}),
            ("/api/targets", with_option({"parallel": "yes"}), None, "invalid_boolean", {"field": "parallel"}),
            ("/api/targets", with_option({"models": ["unknown"]}), None, "invalid_models", {}),
            ("/api/targets", with_option({"seed": -1}), None, "invalid_seed", {}),
            ("/api/targets", with_option({"gpus": [-1]}), None, "invalid_gpus", {}),
            ("/api/targets", with_option({"max_workers": 0}), None, "invalid_max_workers", {}),
            ("/api/targets", {"targets": [{"name": valid["name"]}]}, None, "msa_missing", {"name": valid["name"]}),
            ("/api/import", {"config": {}}, None, "config_samples_required", {}),
            ("/api/import", {"config": {"samples": [{"name": valid["name"], "msa_path": "../qa_missing_alignment.a3m"}]}},
             None, "input_path_outside_project", {"path": "../qa_missing_alignment.a3m"}),
        )
        with mock.patch("trflow.web.subprocess.Popen", side_effect=AssertionError("Invalid inputs cannot start prediction")):
            for endpoint, body, headers, code, params in cases:
                with self.subTest(code=code, body=body):
                    status, error = self.request("POST", endpoint, body, headers)
                    self.assertEqual(status, 400, error)
                    self.assertEqual(error["error_code"], code)
                    if params is None:
                        self.assertEqual(set(error["error_params"]), {"detail"})
                        self.assertTrue(error["error_params"]["detail"])
                    else:
                        self.assertEqual(error["error_params"], params)
        self.assertEqual(self.manager.store.list(), [])
        self.assertEqual(list(self.manager.jobs_dir.iterdir()), [])

    def test_cluster_paths_and_recycle_metadata_errors_are_structured(self):
        record = self.target(status="failed")
        for value in ("abc", "0", "-1", "1.5"):
            with self.subTest(k=value):
                status, error = self.request("GET", f"/api/targets/{record['id']}/analysis?k={value}")
                self.assertEqual(status, 400, error)
                self.assertEqual(error["error_code"], "invalid_cluster_count")
                self.assertEqual(error["error_params"], {})
        paths = (
            ("/static/%2e%2e/web.py", "path_outside_target"),
            (f"/api/targets/{record['id']}/structures/%2e%2e%5cprivate.pdb", "invalid_structure_filename"),
        )
        for path, code in paths:
            with self.subTest(path=path):
                status, error = self.request("GET", path)
                self.assertEqual(status, 400, error)
                self.assertEqual(error["error_code"], code)
        self.manager.delete(record["id"])
        manifest = self.manager.recycle_dir / record["id"] / "manifest.json"
        metadata = read_json(manifest)
        metadata["record"]["id"] = "qa-mismatched-target"
        write_json(manifest, metadata)
        status, error = self.request("POST", f"/api/recycle-bin/{record['id']}/restore", {})
        self.assertEqual(status, 400, error)
        self.assertEqual(error["error_code"], "recycle_metadata_mismatch")
        self.assertEqual(error["error_params"], {})
        self.assertIsNotNone(self.manager.store.get(record["id"], include_deleted=True)["deleted_at"])

    def test_cancel_retry_and_completion_clear_stale_error_metadata(self):
        stale = {"error": INTERRUPTED, "error_code": "service_interrupted", "error_params": {"obsolete": "QA"}}
        cancelled = self.target(**stale)
        self.manager.cancel(cancelled["id"])
        retried = self.target(status="failed", **stale)
        output = Path(retried["output_path"])
        predictions = output / "predictions"
        predictions.mkdir(parents=True)
        for index in range(3):
            (predictions / f"qa_{index}.pdb").write_text("END\n", encoding="utf-8")
        write_json(output / "info.json", {"predictions": [{"file": f"qa_{index}.pdb"} for index in range(3)]})
        self.assertEqual(self.manager.retry(retried["id"])[0]["id"], retried["id"])
        for target_id in (cancelled["id"], retried["id"]):
            stored = self.manager.store.get(target_id)
            self.assertIsNone(stored["error"])
            self.assertIsNone(stored["error_code"])
            self.assertEqual(stored["error_params"], {})
        self.manager.store.update(retried["id"], **stale)
        with mock.patch.object(self.manager, "analysis", return_value={}), \
                mock.patch("trflow.web.subprocess.Popen", side_effect=AssertionError("Completion QA cannot start GPU")):
            self.manager._finish_analysis(self.manager.store.get(retried["id"]))
        completed = self.manager.store.get(retried["id"])
        self.assertEqual(completed["status"], "completed")
        self.assertIsNone(completed["error"])
        self.assertIsNone(completed["error_code"])
        self.assertEqual(completed["error_params"], {})

    def test_worker_clears_stale_metadata_then_persists_the_new_controlled_failure(self):
        record = self.target(error=INTERRUPTED, error_code="service_interrupted", error_params={"obsolete": "QA"})
        entered = []
        manager = self.manager

        class FakeFailedPrediction:
            def __init__(self, command, **kwargs):
                current = manager.store.get(record["id"])
                entered.append(current["status"])
                assert current["error"] is None
                assert current["error_code"] is None
                assert current["error_params"] == {}

            def poll(self):
                return 4

            def wait(self):
                return 4

        def run_once(current):
            try:
                manager._run_locked(current)
            finally:
                manager.stop_event.set()

        with mock.patch.object(manager, "_run", side_effect=run_once), \
                mock.patch("trflow.web.subprocess.Popen", FakeFailedPrediction):
            manager._worker()
        failed = manager.store.get(record["id"])
        self.assertEqual(entered, ["running"])
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error_code"], "prediction_failed")
        self.assertEqual(failed["error_params"], {"exit_code": 4})
        self.assertEqual(error_fields(RAW_DIAGNOSTIC), {"error": RAW_DIAGNOSTIC, "error_code": None, "error_params": {}})
        self.assertIsNone(manager.process)

    def test_every_controlled_backend_code_has_a_frontend_translation(self):
        codes = set(web_errors._MESSAGES.values()) | {code for _, code in web_errors._PATTERNS} | {"invalid_json"}
        keys = set(re.findall(r"[\"']server\.([a-z_]+)[\"']\s*:", (STATIC / "i18n.js").read_text(encoding="utf-8")))
        self.assertFalse(codes - keys, "Missing frontend server translations: " + ", ".join(sorted(codes - keys)))

    def test_http_core_errors_have_codes_and_keep_conflict_compatibility(self):
        record = self.target()
        original = self.manager.store.get(record["id"])
        cases = (
            ("GET", "/api/targets/qa-target-does-not-exist", 404, "target_not_found"),
            ("GET", f"/api/targets/{record['id']}/analysis", 400, "structures_not_ready"),
            ("GET", f"/api/targets/{record['id']}/download", 400, "structures_not_ready"),
            ("DELETE", "/api/targets/" + record["id"], 409, "target_active"),
        )
        with mock.patch("trflow.web.subprocess.Popen", side_effect=AssertionError("Error routes cannot start prediction")):
            for method, path, expected_status, code in cases:
                with self.subTest(method=method, path=path):
                    status, error = self.request(method, path)
                    self.assertEqual(status, expected_status, error)
                    self.assertIsInstance(error["error"], str)
                    self.assertTrue(error["error"])
                    self.assertEqual(error["error_code"], code)
                    self.assertEqual(error["error_params"], {})
                    if expected_status == 409:
                        self.assertEqual(error["code"], code)
        self.assertEqual(self.manager.store.get(record["id"]), original)
        self.assertTrue(Path(record["job_path"]).is_dir())
        self.assertFalse(self.manager.recycle_dir.exists())
        self.assertIsNone(self.manager.process)


if __name__ == "__main__":
    unittest.main()
