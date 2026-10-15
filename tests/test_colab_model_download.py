"""Offline Colab download contracts; no model loading, network or GPU needed."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "colab" / "model_download.py"
spec = importlib.util.spec_from_file_location("colab_download_under_test", MODULE_PATH)
wd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wd)


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.payloads = {name: ("small-fixture:" + name).encode() for name in wd.MODEL_FILES}
        self.manifest = {
            name: {"id": "public-file-" + str(index), "bytes": len(data),
                   "sha256": hashlib.sha256(data).hexdigest()}
            for index, (name, data) in enumerate(self.payloads.items())
        }
        manifest_patch = patch.object(wd, "MODEL_FILES", self.manifest)
        manifest_patch.start()
        self.addCleanup(manifest_patch.stop)
        self.calls = []
        self.fallback = Mock(side_effect=AssertionError("unexpected fallback"))

    def download(self, **kwargs):
        self.calls.append(kwargs)
        name = next(name for name, row in self.manifest.items() if row["id"] == kwargs["id"])
        data = self.payloads[name]
        kwargs["progress"](len(data), len(data))
        Path(kwargs["output"]).write_bytes(data)
        return kwargs["output"]

    def ensure(self, **kwargs):
        kwargs.setdefault("downloader", self.download)
        return wd.ensure_model_weights(self.cache, **kwargs)

    def folder(self, name="legacy"):
        directory = self.root / name
        directory.mkdir()
        for filename, data in self.payloads.items():
            (directory / filename).write_bytes(data)
        return directory

    def receipt(self):
        return json.loads((self.cache / "current.json").read_text(encoding="utf-8"))

    def identity(self):
        return hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode()).hexdigest()[:16]

    def partial(self):
        directory = self.cache / ("partial-" + self.identity())
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def assert_no_receipt(self):
        self.assertFalse((self.cache / "current.json").exists())

    def test_cold_download_uses_only_allowlisted_ids_and_safe_options(self):
        result = self.ensure()
        self.assertEqual(wd.validate_weights(result), result)
        self.assertEqual([call["id"] for call in self.calls],
                         [row["id"] for row in self.manifest.values()])
        for call in self.calls:
            self.assertFalse(call["use_cookies"])
            self.assertTrue(call["verify"])
            self.assertTrue(call["resume"])
            self.assertEqual(call["timeout"], (15, 60))
            self.assertEqual(call["retries"], 2)
        record = self.receipt()
        self.assertEqual(record["source"], "Google Drive")
        self.assertEqual(record["files"], self.manifest)
        self.assertEqual(Path(record["directory"]), result)
        self.assertEqual(list(self.cache.glob(".receipt-*")), [])

    def test_warm_cache_downloads_nothing_and_revalidates_hashes(self):
        result = self.ensure()
        self.calls.clear()
        with patch.object(wd, "_default_downloader", side_effect=AssertionError("network")):
            self.assertEqual(wd.ensure_model_weights(self.cache), result)
        self.assertEqual(self.calls, [])

    def test_corrupted_completed_file_is_redownloaded_alone(self):
        result = self.ensure()
        name = next(iter(self.payloads))
        (result / name).write_bytes(b"x" * len(self.payloads[name]))
        self.calls.clear()
        self.assertEqual(self.ensure(), result)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["id"], self.manifest[name]["id"])
        self.assertEqual((result / name).read_bytes(), self.payloads[name])

    def test_existing_runtime_cache_is_reused_without_downloading(self):
        legacy = self.folder()
        result = self.ensure(legacy_directory=legacy)
        self.assertEqual(result, legacy)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.receipt()["source"], "Existing runtime cache")

    def test_invalid_legacy_cache_does_not_hide_public_download(self):
        legacy = self.folder()
        (legacy / next(iter(self.payloads))).write_bytes(b"bad")
        result = self.ensure(legacy_directory=legacy)
        self.assertNotEqual(result, legacy)
        self.assertEqual(len(self.calls), 4)

    def test_drive_failure_uses_verified_author_fallback(self):
        alternative = self.folder("author")
        fallback = Mock(return_value=alternative)
        failed = Mock(side_effect=OSError("network unavailable"))
        self.assertEqual(self.ensure(downloader=failed, fallback=fallback), alternative)
        fallback.assert_called_once_with()
        self.assertEqual(self.receipt()["source"], "Author server")

    def test_bad_fallback_is_not_published(self):
        alternative = self.folder("author")
        (alternative / next(iter(self.payloads))).write_bytes(b"bad")
        with self.assertRaisesRegex(wd.ModelDownloadError, "could not finish"):
            self.ensure(downloader=Mock(return_value=None), fallback=lambda: alternative)
        self.assert_no_receipt()

    def test_fallback_failure_does_not_discard_successful_drive_files(self):
        def fail_second(**kwargs):
            if self.calls:
                raise OSError("offline")
            return self.download(**kwargs)
        with self.assertRaises(wd.ModelDownloadError):
            self.ensure(downloader=fail_second, fallback=Mock(side_effect=OSError("offline")))
        self.assert_no_receipt()
        self.calls.clear()
        result = self.ensure()
        self.assertEqual(len(self.calls), 3)
        wd.validate_weights(result)

    def test_keyboard_interrupt_never_starts_fallback(self):
        with self.assertRaises(KeyboardInterrupt):
            self.ensure(downloader=Mock(side_effect=KeyboardInterrupt), fallback=self.fallback)
        self.fallback.assert_not_called()
        self.assert_no_receipt()

    def test_system_exit_never_starts_fallback(self):
        with self.assertRaises(SystemExit):
            self.ensure(downloader=Mock(side_effect=SystemExit), fallback=self.fallback)
        self.fallback.assert_not_called()

    def test_resume_keeps_partial_bytes_and_skips_completed_files(self):
        def interrupted(**kwargs):
            if not self.calls:
                return self.download(**kwargs)
            Path(kwargs["output"] + "fixture.part").write_bytes(b"retained bytes")
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.ensure(downloader=interrupted, fallback=self.fallback)
        self.assert_no_receipt()
        self.calls.clear()
        second = list(self.payloads)[1]
        def resumed(**kwargs):
            if kwargs["id"] == self.manifest[second]["id"]:
                partial = Path(kwargs["output"] + "fixture.part")
                self.assertEqual(partial.read_bytes(), b"retained bytes")
                self.assertTrue(kwargs["resume"])
                partial.unlink()
            return self.download(**kwargs)
        wd.validate_weights(self.ensure(downloader=resumed))
        self.assertEqual(len(self.calls), 3)
        self.fallback.assert_not_called()

    def test_invalid_final_download_is_quarantined_before_retry(self):
        incoming = self.partial()
        name = next(iter(self.payloads))
        output = incoming / (name + ".download")
        output.write_bytes(b"interrupted bad output")
        def checked(**kwargs):
            self.assertFalse(Path(kwargs["output"]).exists())
            return self.download(**kwargs)
        wd.validate_weights(self.ensure(downloader=checked))
        quarantined = list(incoming.glob(name + ".invalid-*"))
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0].read_bytes(), b"interrupted bad output")

    def test_valid_final_download_can_be_reused_by_resume(self):
        incoming = self.partial()
        name = next(iter(self.payloads))
        output = incoming / (name + ".download")
        output.write_bytes(self.payloads[name])
        def skip_existing(**kwargs):
            if Path(kwargs["output"]).exists():
                return kwargs["output"]
            return self.download(**kwargs)
        wd.validate_weights(self.ensure(downloader=skip_existing))
        self.assertEqual(len(self.calls), 3)

    def test_wrong_size_or_hash_rejected_before_receipt(self):
        for data in (b"too short", b"x" * len(next(iter(self.payloads.values())))):
            with self.subTest(size=len(data)):
                def bad(**kwargs):
                    Path(kwargs["output"]).write_bytes(data)
                    return kwargs["output"]
                with self.assertRaisesRegex(wd.ModelDownloadError, "integrity"):
                    self.ensure(downloader=bad)
                self.assert_no_receipt()

    def test_none_result_is_failure_even_if_bytes_exist(self):
        def no_result(**kwargs):
            self.download(**kwargs)
            return None
        with self.assertRaises(wd.ModelDownloadError):
            self.ensure(downloader=no_result)
        self.assert_no_receipt()

    def test_progress_rejects_unreviewed_lengths(self):
        first_size = next(iter(self.manifest.values()))["bytes"]
        for received, announced in ((-1, None), (first_size + 1, None),
                                     (0, first_size + 1), ("0", None)):
            with self.subTest(received=received, announced=announced):
                def bad(**kwargs):
                    kwargs["progress"](received, announced)
                with self.assertRaises(wd.ModelDownloadError):
                    self.ensure(downloader=bad)
                self.assert_no_receipt()

    def test_unknown_content_length_is_allowed_if_final_hash_matches(self):
        def unknown(**kwargs):
            kwargs["progress"](0, None)
            return self.download(**kwargs)
        wd.validate_weights(self.ensure(downloader=unknown))

    def test_progress_rejects_transfer_over_time_limit(self):
        with patch.object(wd, "FILE_TIMEOUT", -1):
            with self.assertRaises(wd.ModelDownloadError):
                self.ensure()
        self.assert_no_receipt()

    def test_disk_space_checked_before_network(self):
        with patch.object(wd.shutil, "disk_usage", return_value=Mock(free=0)):
            with self.assertRaisesRegex(wd.ModelDownloadError, "storage"):
                self.ensure(fallback=self.fallback)
        self.assertEqual(self.calls, [])
        self.fallback.assert_not_called()

    def test_disk_reserve_checked_during_transfer(self):
        sufficient = Mock(free=10**12)
        exhausted = Mock(free=0)
        with patch.object(wd.shutil, "disk_usage", side_effect=[sufficient, exhausted]):
            with self.assertRaises(wd.ModelDownloadError):
                self.ensure()
        self.assert_no_receipt()

    def test_extra_files_in_cache_are_not_silently_accepted(self):
        result = self.ensure()
        (result / "unexpected.txt").write_text("unexpected", encoding="utf-8")
        self.calls.clear()
        with self.assertRaises(wd.ModelCacheError):
            self.ensure(fallback=self.fallback)
        self.assertEqual(self.calls, [])
        self.fallback.assert_not_called()

    def test_manifest_change_uses_separate_cache_generation(self):
        old = self.ensure()
        name = next(iter(self.payloads))
        self.payloads[name] += b"new generation"
        self.manifest[name]["bytes"] = len(self.payloads[name])
        self.manifest[name]["sha256"] = hashlib.sha256(self.payloads[name]).hexdigest()
        self.calls.clear()
        new = self.ensure()
        self.assertNotEqual(old, new)
        self.assertTrue(old.is_dir())
        self.assertEqual(len(self.calls), 4)
        wd.validate_weights(new)

    def test_validate_weights_requires_exact_names(self):
        directory = self.folder()
        (directory / "unreviewed.pth").write_bytes(b"extra")
        with self.assertRaises(wd.ModelDownloadError):
            wd.validate_weights(directory)

    def test_model_path_cannot_be_a_directory(self):
        incoming = self.partial()
        (incoming / (next(iter(self.payloads)) + ".download")).mkdir()
        with self.assertRaises(wd.ModelCacheError):
            self.ensure(fallback=self.fallback)
        self.fallback.assert_not_called()

    def test_cache_safety_errors_never_trigger_fallback(self):
        with self.assertRaises(wd.ModelCacheError):
            self.ensure(downloader=Mock(side_effect=wd.ModelCacheError("unsafe")),
                        fallback=self.fallback)
        self.fallback.assert_not_called()

    def test_receipt_atomic_replace_failure_cleans_temporary_file(self):
        directory = self.folder()
        self.cache.mkdir()
        with patch.object(wd.os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaises(OSError):
                wd._record(self.cache, directory, "test")
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_linked_cache_partial_or_receipt_is_rejected(self):
        target = self.root / "target"
        target.mkdir()
        link = self.root / "link"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("OS does not permit test symlinks")
        with self.assertRaises(wd.ModelCacheError):
            wd.ensure_model_weights(link / "nested", downloader=self.download,
                                    fallback=self.fallback)
        incoming = self.partial()
        name = next(iter(self.payloads))
        (incoming / (name + ".downloadfixture.part")).symlink_to(target,
                                                                  target_is_directory=True)
        with self.assertRaises(wd.ModelCacheError):
            self.ensure(fallback=self.fallback)
        (self.cache / "current.json").symlink_to(target, target_is_directory=True)
        with self.assertRaises(wd.ModelCacheError):
            self.ensure(fallback=self.fallback)
        self.fallback.assert_not_called()


class NotebookDownloadContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((ROOT / "trFlow.ipynb").read_text(encoding="utf-8"))
        cls.code = ["".join(cell["source"]) for cell in cls.notebook["cells"]
                    if cell["cell_type"] == "code"]
        cls.setup = cls.code[0]

    def assignment(self, source, name):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                                                    and target.id == name
                                                    for target in node.targets):
                return ast.literal_eval(node.value)
        self.fail("Missing assignment: " + name)

    def test_all_code_and_embedded_helpers_compile_without_execution(self):
        for index, source in enumerate(self.code):
            with self.subTest(cell=index):
                tree = ast.parse(source)
                compile(tree, "notebook-cell-" + str(index), "exec")
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id in {
                                "STATUS_SOURCE", "HELPER_SOURCE", "USER_UI_SOURCE",
                                "WEIGHT_DOWNLOAD_SOURCE", "VIEWER_SOURCE",
                            }:
                                embedded = ast.literal_eval(node.value)
                                compile(embedded, target.id, "exec")

    def test_embedded_downloader_matches_reviewed_module_verbatim(self):
        self.assertEqual(self.assignment(self.setup, "WEIGHT_DOWNLOAD_SOURCE"),
                         MODULE_PATH.read_text(encoding="utf-8"))

    def test_drive_and_author_manifests_pin_identical_checkpoint_bytes(self):
        helper = self.assignment(self.setup, "HELPER_SOURCE")
        author = self.assignment(helper, "OFFICIAL_WEIGHT_MANIFEST")
        self.assertEqual(set(author), set(wd.MODEL_FILES))
        for name in author:
            with self.subTest(file=name):
                self.assertEqual(author[name]["sha256"], wd.MODEL_FILES[name]["sha256"])
                self.assertEqual(author[name]["bytes"], wd.MODEL_FILES[name]["bytes"])

    def test_single_setup_step_with_no_source_selector_or_drive_mount(self):
        all_text = "\n".join("".join(cell["source"]) for cell in self.notebook["cells"])
        self.assertTrue(self.setup.startswith("# @title Step 1"))
        self.assertNotIn("Step 1a", all_text)
        self.assertNotIn("Step 1b", all_text)
        self.assertNotIn("MODEL_SOURCE", all_text)
        self.assertNotIn("drive.mount", all_text)
        self.assertNotIn("choose My Drive", all_text)
        self.assertIn("ensure_model_weights", self.setup)
        self.assertIn("fallback=lambda:h.ensure_official_weights", self.setup)

    def test_readiness_is_cleared_before_setup_can_fail(self):
        self.assertLess(self.setup.index("MODEL_WEIGHT_SETTINGS = None"),
                        self.setup.index("STATUS_SOURCE"))
        self.assertLess(self.setup.index("MODEL_FOLDER = None"),
                        self.setup.index("STATUS_SOURCE"))
        self.assertLess(self.setup.index("MODEL_FOLDER=wd.ensure_model_weights"),
                        self.setup.index("MODEL_WEIGHT_SETTINGS=dict"))

    def test_published_notebook_contains_no_execution_outputs(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])


if __name__ == "__main__":
    unittest.main()
