"""Portable server/filesystem checks; all state belongs to disposable fixtures."""
from __future__ import annotations

import http.client
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from trflow.web import JobManager, REPO, create_handler, default_data_dir


class PortableWebTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trflow portable α 测试 ")
        self.root = Path(self.temporary.name)
        self.manager = JobManager(self.root / "queue data", import_existing=False, start_worker=False)

    def tearDown(self):
        self.manager.close()
        self.temporary.cleanup()

    def test_real_loopback_server_under_unicode_spaced_path_and_stable_mime(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(self.manager))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with mock.patch("trflow.web.mimetypes.guess_type", return_value=("application/broken-registry", None)), \
                    mock.patch("trflow.web.subprocess.Popen", side_effect=AssertionError("No prediction subprocess in portable QA")):
                for resource, mime in (("/", "text/html"), ("/static/app.js", "text/javascript"),
                                       ("/static/i18n.js", "text/javascript"), ("/static/styles.css", "text/css"),
                                       ("/static/ensemble-hero.svg", "image/svg+xml"),
                                       ("/static/vendor/3Dmol-min.js", "text/javascript"),
                                       ("/api/config", "application/json")):
                    with self.subTest(resource=resource):
                        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                        try:
                            connection.request("GET", resource)
                            response = connection.getresponse()
                            self.assertEqual(response.status, 200)
                            self.assertEqual(response.getheader("Content-Type").split(";")[0], mime)
                            body = response.read()
                            self.assertTrue(body)
                            if resource == "/api/config":
                                self.assertEqual(json.loads(body)["device"], "local")
                        finally:
                            connection.close()
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                try:
                    connection.request("GET", "/api/example")
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    example = json.loads(response.read())
                    self.assertEqual(example["name"], "2akl")
                    self.assertEqual(example["msa_filename"], "2akl.a3m")
                    self.assertTrue(example["msa_text"].startswith(">"))
                finally:
                    connection.close()
            self.assertEqual(self.manager.store.list(), [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_platform_service_lock_is_exclusive_and_released(self):
        with self.assertRaises(RuntimeError):
            JobManager(self.root / "queue data", import_existing=False, start_worker=False)
        self.manager.close()
        replacement = JobManager(self.root / "queue data", import_existing=False, start_worker=False)
        replacement.close()

    def test_windows_reserved_names_remain_display_names_but_use_safe_files(self):
        names = ["CON", "prn", "AUX", "NUL", "COM1", "COM9", "COM¹", "LPT2", "LPT³", "蛋白 target"]
        with mock.patch("trflow.web.subprocess.Popen", side_effect=AssertionError("No GPU prediction")):
            records = self.manager.submit([{"name": name, "msa_text": ">query\nACDEFG\n"} for name in names])
        for name, public in zip(names, records):
            with self.subTest(name=name):
                record = self.manager.store.get(public["id"])
                self.assertEqual(record["name"], name)
                if name != "蛋白 target":
                    self.assertEqual(record["sample_name"], "target_" + name)
                inputs = Path(record["job_path"]) / "inputs"
                self.assertEqual((inputs / "alignment.a3m").read_text(encoding="utf-8"), ">query\nACDEFG\n")
                output = Path(record["output_path"])
                output.mkdir(parents=True, exist_ok=True)
                proof = output / "portable file.txt"
                proof.write_text("CPU fixture only", encoding="utf-8")
                self.assertEqual(proof.read_text(encoding="utf-8"), "CPU fixture only")

    def test_default_data_directory_distinguishes_checkout_and_install(self):
        self.assertEqual(default_data_dir(), REPO / "outputs" / "web")
        with mock.patch("trflow.web.REPO", self.root / "installed package"), \
                mock.patch("trflow.web.Path.cwd", return_value=self.root / "working directory"):
            self.assertEqual(default_data_dir(), self.root / "working directory" / "outputs" / "web")

    def test_analysis_worker_uses_utf8_and_preserves_unicode_diagnostics(self):
        public = self.manager.submit([{"name": "蛋白 📦", "msa_text": ">query\nACDEFG\n",
                                       "options": {"sample_num": 1}}])[0]
        record = self.manager.store.update(public["id"], status="completed", generated=1)
        diagnostic = "RuntimeError: 无法分析模型 蛋白 📦.pdb\n  原始诊断应保留\n"
        process = mock.Mock(returncode=3)
        process.communicate.return_value = (diagnostic.encode("utf-8"), None)
        with mock.patch("trflow.web.subprocess.Popen", return_value=process) as spawn:
            with self.assertRaises(RuntimeError) as caught:
                self.manager.analysis(record["id"])
        self.assertEqual(str(caught.exception), "结构分析失败：" + diagnostic)
        environment = spawn.call_args.kwargs["env"]
        self.assertEqual(environment["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(environment["PYTHONUNBUFFERED"], "1")
        self.assertEqual(spawn.call_args.kwargs["start_new_session"], os.name != "nt")
        self.assertEqual((Path(record["job_path"]) / "analysis" / "analysis.log").read_bytes(), diagnostic.encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
