"""Single-chain inputs, bounded public-MSA protocol and notebook UI contracts."""
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib import error, parse


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "colab" / "input_msa.py"
spec = importlib.util.spec_from_file_location("colab_input_under_test", MODULE_PATH)
im = importlib.util.module_from_spec(spec)
spec.loader.exec_module(im)
QUERY = "ACDEFGHIK"


def archive_bytes(entries=None):
    entries = entries if entries is not None else {
        im.RESULT_MEMBERS[0]: ">101\n" + QUERY + "\n>hit1\nACDEYGHIK\n",
        im.RESULT_MEMBERS[1]: ">101\n" + QUERY + "\n>hit1-again\nACDEYGHIK\n>hit2\nACdDEFGH-K\n\x00",
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, text in entries.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return stream.getvalue()


class FakeService:
    def __init__(self, statuses=None, data=None):
        self.calls = []
        self.statuses = list(statuses or ["COMPLETE"])
        self.archive = archive_bytes() if data is None else data

    def __call__(self, method, route, **kwargs):
        self.calls.append((method, route, kwargs))
        if method == "POST":
            return json.dumps({"status": "PENDING", "id": "fixture-ticket"}).encode()
        if route.startswith("result/download/"):
            return self.archive
        return json.dumps({"status": self.statuses.pop(0) if self.statuses else "COMPLETE"}).encode()


class InputTests(unittest.TestCase):
    def test_sequence_whitespace_and_lowercase_are_normalized(self):
        self.assertEqual(im.normalize_sequence("  acd ef\ngHik\t"), QUERY)

    def test_one_fasta_record_is_accepted(self):
        self.assertEqual(im.normalize_sequence("\ufeff>public example\nACDE\nFGHIK\n"), QUERY)

    def test_multiple_fasta_records_are_never_concatenated(self):
        with self.assertRaisesRegex(im.MSAError, "exactly one"):
            im.normalize_sequence(">one\nACDE\n>two\nFGHIK\n")

    def test_invalid_targets_are_rejected(self):
        for text in ("", ">only header", ">\nACD", "ACD:EF", "ACD-EF", "ACD*EF", "ACD123",
                     "ACD.EF", "ACDß", "ACDBZ", "ACDΩ"):
            with self.subTest(text=text), self.assertRaises(im.MSAError):
                im.normalize_sequence(text)

    def test_sequence_limits_and_input_type(self):
        for text in (None, "A" * (im.MAX_SEQUENCE + 1)):
            with self.subTest(kind=type(text).__name__), self.assertRaises(im.MSAError):
                im.normalize_sequence(text)
        with patch.object(im, "MAX_INPUT_BYTES", 3), self.assertRaises(im.MSAError):
            im.normalize_sequence(QUERY)

    def test_a3m_preserves_lowercase_insertions_and_checks_columns(self):
        query, records = im.parse_a3m(">query\n" + QUERY + "\n>homolog\nACdDEFGH-K\n", QUERY)
        self.assertEqual(query, QUERY)
        self.assertEqual(records[1][1], "ACdDEFGH-K")

    def test_mmseqs2_tab_separated_hit_metadata_is_valid(self):
        _, records = im.parse_a3m(">101\n" + QUERY + "\n>UniRef100_example\t193\t0.982\t1.559E-53\nACDEYGHIK\n", QUERY)
        self.assertEqual(records[1][0], "UniRef100_example\t193\t0.982\t1.559E-53")

    def test_a3m_rejects_invalid_query_and_alignment_records(self):
        for text in ("#9,1\n>query\n" + QUERY, ">q\nacdefghik", ">q\nACDE-GHIK",
                     ">q\n" + QUERY + "\n>bad\nACD", ">q\n" + QUERY + "\n>bad\nACD.EGHIK",
                     ">q\n" + QUERY + "\n>empty", ">\n" + QUERY):
            with self.subTest(text=text), self.assertRaises(im.MSAError):
                im.parse_a3m(text)

    def test_a3m_rejects_wrong_query_record_count_and_size(self):
        with self.assertRaises(im.MSAError):
            im.parse_a3m(">q\nACDEFGHIK", "ACDEYGHIK")
        with patch.object(im, "MAX_RECORDS", 1), self.assertRaises(im.MSAError):
            im.parse_a3m(">q\n" + QUERY + "\n>hit\n" + QUERY)
        with patch.object(im, "MAX_A3M_BYTES", 4), self.assertRaises(im.MSAError):
            im.parse_a3m(">q\n" + QUERY)


class MSAClientTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.service = FakeService()
        self.elapsed = 0
        self.messages = []

    def sleep(self, duration):
        self.elapsed += duration

    def ensure(self, **kwargs):
        kwargs.setdefault("allow_public", True)
        kwargs.setdefault("requester", self.service)
        kwargs.setdefault("sleep", self.sleep)
        kwargs.setdefault("clock", lambda: self.elapsed)
        kwargs.setdefault("progress", self.messages.append)
        return im.ensure_msa(QUERY, self.cache, **kwargs)

    def test_cold_query_uses_official_msa_protocol_without_predictions(self):
        result = self.ensure()
        query, rows = im.parse_a3m(result.read_text(), QUERY)
        self.assertEqual(query, QUERY)
        self.assertEqual(len(rows), 3)  # Query and repeated hits are deduplicated.
        self.assertEqual([call[:2] for call in self.service.calls], [
            ("POST", "ticket/msa"), ("GET", "ticket/fixture-ticket"),
            ("GET", "result/download/fixture-ticket")])
        self.assertEqual(self.service.calls[0][2]["fields"], {"q": ">101\n" + QUERY + "\n", "mode": "env"})
        receipt = json.loads((result.parent / "result.json").read_text())
        self.assertEqual(receipt["sha256"], hashlib.sha256(result.read_bytes()).hexdigest())
        self.assertEqual(receipt["records"], 3)
        self.assertEqual(list(result.parent.glob(".msa-*")), [])

    def test_consent_required_before_any_network_call(self):
        for consent in (False, None, 1, "yes"):
            with self.subTest(consent=consent), self.assertRaises(im.MSAError):
                self.ensure(allow_public=consent)
        self.assertEqual(self.service.calls, [])

    def test_warm_cache_can_be_reused_without_public_consent_or_network(self):
        result = self.ensure()
        self.service.calls.clear()
        self.assertEqual(self.ensure(allow_public=False), result)
        self.assertEqual(self.service.calls, [])
        self.assertIn("Using the saved, verified MSA", self.messages)

    def test_corrupt_alignment_is_retrieved_again_without_resubmission(self):
        result = self.ensure()
        result.write_text(">wrong\nACDEYGHIK\n")
        self.service.calls.clear()
        self.ensure()
        self.assertTrue(all(method != "POST" for method, _, _ in self.service.calls))
        im.parse_a3m(result.read_text(), QUERY)

    def test_pending_statuses_are_polled_serially(self):
        self.service.statuses = ["PENDING", "RUNNING", "UNKNOWN", "COMPLETE"]
        self.ensure()
        self.assertEqual(self.elapsed, 3 * im.POLL_SECONDS)
        self.assertEqual(sum(method == "POST" for method, _, _ in self.service.calls), 1)

    def test_cancelled_search_retains_ticket_and_resumes_without_post(self):
        self.service.statuses = ["RUNNING"]
        with self.assertRaises(KeyboardInterrupt):
            self.ensure(sleep=Mock(side_effect=KeyboardInterrupt))
        self.assertEqual(len(list(self.cache.rglob("ticket.json"))), 1)
        self.assertEqual(list(self.cache.rglob("alignment.a3m")), [])
        self.service.calls.clear()
        self.ensure()
        self.assertFalse(any(method == "POST" for method, _, _ in self.service.calls))

    def test_old_ticket_is_not_polled_without_consent(self):
        self.service.statuses = ["RUNNING"]
        with self.assertRaises(KeyboardInterrupt):
            self.ensure(sleep=Mock(side_effect=KeyboardInterrupt))
        self.service.calls.clear()
        with self.assertRaises(im.MSAError):
            self.ensure(allow_public=False)
        self.assertEqual(self.service.calls, [])

    def test_timeout_is_bounded_and_ticket_can_resume(self):
        self.service.statuses = ["RUNNING"] * 10
        with patch.object(im, "MAX_WAIT_SECONDS", 10), self.assertRaisesRegex(im.MSAError, "timed out"):
            self.ensure()
        self.assertEqual(list(self.cache.rglob("alignment.a3m")), [])
        self.assertEqual(len(list(self.cache.rglob("ticket.json"))), 1)

    def test_rate_limited_submissions_have_bounded_backoff(self):
        busy = Mock(return_value=b'{"status":"RATELIMIT"}')
        with self.assertRaises(im.MSAError):
            self.ensure(requester=busy)
        self.assertEqual(busy.call_count, im.MAX_SUBMISSIONS)
        self.assertEqual(self.elapsed, 60)

    def test_transient_network_errors_are_bounded(self):
        offline = Mock(side_effect=error.URLError("offline"))
        with self.assertRaises(im.MSAError):
            self.ensure(requester=offline)
        self.assertEqual(offline.call_count, im.MAX_REQUEST_RETRIES)
        self.assertEqual(self.elapsed, 15)
        self.assertEqual(list(self.cache.rglob("alignment.a3m")), [])

    def test_http_429_is_retried_but_redirect_is_not_followed(self):
        for code, expected_calls in ((429, im.MAX_REQUEST_RETRIES), (302, 1), (403, 1)):
            with self.subTest(code=code):
                failed = Mock(side_effect=error.HTTPError(im.API_URL, code, "service", {}, None))
                with self.assertRaises(im.MSAError):
                    self.ensure(requester=failed)
                self.assertEqual(failed.call_count, expected_calls)

    def test_invalid_json_or_job_status_is_not_published(self):
        for data in (b"not json", b"[]", b"{}", b'{"status":true}'):
            with self.subTest(data=data), self.assertRaises(im.MSAError):
                self.ensure(requester=Mock(return_value=data))
        self.assertEqual(list(self.cache.rglob("alignment.a3m")), [])

    def test_invalid_ticket_cannot_change_api_route(self):
        for ticket in ("../other", "x?sequence=secret", "https://elsewhere", "", 123):
            response = json.dumps({"status": "PENDING", "id": ticket}).encode()
            sender = Mock(return_value=response)
            with self.subTest(ticket=ticket), self.assertRaises(im.MSAError):
                self.ensure(requester=sender)
            self.assertEqual(sender.call_count, 1)

    def test_failed_ticket_is_not_reused_forever(self):
        self.service.statuses = ["ERROR"]
        with self.assertRaises(im.MSAError):
            self.ensure()
        self.service.calls.clear()
        self.ensure()
        self.assertEqual(self.service.calls[0][0], "POST")

    def test_maintenance_and_unknown_terminal_status_are_reported(self):
        for status in ("MAINTENANCE", "ERROR", "UNEXPECTED"):
            sender = Mock(return_value=json.dumps({"status": status}).encode())
            with self.subTest(status=status), self.assertRaises(im.MSAError):
                self.ensure(requester=sender)
            self.assertEqual(sender.call_count, 1)

    def test_service_failure_never_fabricates_query_only_alignment(self):
        self.service.archive = b"not an archive"
        with self.assertRaises(im.MSAError):
            self.ensure()
        self.assertEqual(list(self.cache.rglob("alignment.a3m")), [])

    def test_changed_sequence_or_mode_uses_separate_cache(self):
        first = self.ensure()
        with patch.object(im, "API_MODE", "all"):
            with self.assertRaises(im.MSAError):
                self.ensure(allow_public=False)
        self.assertEqual(len(list(self.cache.iterdir())), 2)
        self.assertTrue(first.is_file())

    def test_receipt_does_not_expose_query_sequence_or_fasta_header(self):
        result = self.ensure()
        for path in result.parent.glob("*.json"):
            self.assertNotIn(QUERY, path.read_text())

    def test_unsafe_cache_file_is_rejected_before_network(self):
        result = self.ensure()
        result.unlink()
        result.mkdir()
        self.service.calls.clear()
        with self.assertRaises(im.MSAError):
            self.ensure()
        self.assertEqual(self.service.calls, [])


class ArchiveAndTransportTests(unittest.TestCase):
    def test_missing_member_and_mismatched_query_are_rejected(self):
        for entries in ({im.RESULT_MEMBERS[0]: ">q\n" + QUERY},
                        {name: ">wrong\nACDEYGHIK\n" for name in im.RESULT_MEMBERS}):
            with self.subTest(names=list(entries)), self.assertRaises(im.MSAError):
                im._alignment_from_archive(archive_bytes(entries), QUERY, lambda: None)

    def test_unsafe_paths_and_symlinks_are_never_extracted(self):
        for name in ("../uniref.a3m", "/uniref.a3m", "C:/uniref.a3m", "x\\uniref.a3m"):
            with self.subTest(name=name), self.assertRaises(im.MSAError):
                im._alignment_from_archive(archive_bytes({name: "bad"}), QUERY, lambda: None)
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            link = tarfile.TarInfo("uniref.a3m")
            link.type, link.linkname = tarfile.SYMTYPE, "/outside"
            archive.addfile(link)
        with self.assertRaises(im.MSAError):
            im._alignment_from_archive(stream.getvalue(), QUERY, lambda: None)

    def test_duplicate_members_and_expansion_limits_are_rejected(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for _ in range(2):
                entry = tarfile.TarInfo(im.RESULT_MEMBERS[0])
                entry.size = 3
                archive.addfile(entry, io.BytesIO(b"bad"))
        with self.assertRaises(im.MSAError):
            im._alignment_from_archive(stream.getvalue(), QUERY, lambda: None)
        with patch.object(im, "MAX_EXPANDED_BYTES", 4), self.assertRaises(im.MSAError):
            im._alignment_from_archive(archive_bytes(), QUERY, lambda: None)
        with patch.object(im, "MAX_ARCHIVE_BYTES", 4), self.assertRaises(im.MSAError):
            im._alignment_from_archive(archive_bytes(), QUERY, lambda: None)

    def test_archive_loop_obeys_cancellation(self):
        with self.assertRaises(KeyboardInterrupt):
            im._alignment_from_archive(archive_bytes(), QUERY, Mock(side_effect=KeyboardInterrupt))

    def response(self, data, announced=None):
        response = Mock()
        response.status, response.geturl.return_value = 200, im.API_URL + "/ticket/msa"
        response.headers = {} if announced is None else {"Content-Length": announced}
        response.read.side_effect = io.BytesIO(data).read
        manager = Mock()
        manager.__enter__ = Mock(return_value=response)
        manager.__exit__ = Mock(return_value=False)
        return manager

    def test_transport_sends_only_public_msa_form_and_tool_user_agent(self):
        opener = Mock()
        opener.open.return_value = self.response(b'{"status":"PENDING"}')
        fields = {"q": ">101\n" + QUERY + "\n", "mode": "env"}
        with patch.object(im.urllib_request, "build_opener", return_value=opener):
            im._http_request("POST", "ticket/msa", fields=fields, limit=1024, timeout=30)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, im.API_URL + "/ticket/msa")
        self.assertEqual(parse.parse_qs(request.data.decode()), {key: [value] for key, value in fields.items()})
        self.assertEqual(request.get_header("User-agent"), im.USER_AGENT)
        self.assertIsNone(im._NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere"))

    def test_transport_rejects_bad_lengths_incomplete_and_oversized_bodies(self):
        for data, announced, limit in ((b"x", "-1", 100), (b"x", "1000", 100),
                                       (b"x", "2", 100), (b"12345", None, 4)):
            opener = Mock()
            opener.open.return_value = self.response(data, announced)
            with self.subTest(announced=announced), patch.object(im.urllib_request, "build_opener", return_value=opener):
                with self.assertRaises(im.MSAError):
                    im._http_request("POST", "ticket/msa", limit=limit, timeout=30)


class NotebookInputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((ROOT / "trFlow.ipynb").read_text(encoding="utf-8"))
        cls.setup = "".join(cls.notebook["cells"][2]["source"])

    def source_value(self, code, name):
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return ast.literal_eval(node.value)
        self.fail("Missing source: " + name)

    def run_preflight(self, mode, *, text="", upload=None, consent=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        colab = ModuleType("google.colab")
        colab.files = SimpleNamespace(upload=Mock(return_value=upload or {}))
        google = ModuleType("google")
        google.colab = colab
        context = {"ROOT": Path(temporary.name), "Path": Path, "sys": sys,
                   "importlib": importlib, "INPUT_MODE": mode,
                   "QUERY_SEQUENCE": text, "ALLOW_PUBLIC_MSA": consent,
                   "_cell_status": Mock()}
        body = next(node.body for node in ast.parse(self.setup).body if isinstance(node, ast.Try))
        preflight = []
        for node in body:
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and node.value.args and isinstance(node.value.args[0], ast.Constant)
                    and node.value.args[0].value == "Checking GPU and workspace"):
                break
            preflight.append(node)
        with patch.dict(sys.modules, {"google": google, "google.colab": colab}):
            exec(compile(ast.Module(body=preflight, type_ignores=[]), "step1-input-preflight", "exec"), context)
        return context, colab.files.upload

    def test_actual_step1_preflight_accepts_pasted_sequence_before_gpu_setup(self):
        context, upload = self.run_preflight("Paste sequence", text="acde fghik", consent=True)
        self.assertEqual(context["_input_query"], QUERY)
        self.assertEqual(context["_selected_a3m"], "")
        upload.assert_not_called()
        with self.assertRaisesRegex(ValueError, "did not finish"):
            context["read_input_settings"]()

    def test_actual_step1_preflight_accepts_one_fasta_upload(self):
        context, upload = self.run_preflight("Upload FASTA", upload={"protein.fa": (">public\n" + QUERY).encode()}, consent=True)
        self.assertEqual(context["_input_query"], QUERY)
        self.assertTrue(Path(context["_uploaded_fasta"]).is_file())
        upload.assert_called_once_with()

    def test_actual_step1_uploaded_a3m_needs_no_public_consent(self):
        context, _ = self.run_preflight("Upload A3M", upload={"protein.a3m": (">query\n" + QUERY).encode()})
        self.assertEqual(context["_input_query"], QUERY)
        self.assertTrue(Path(context["_selected_a3m"]).is_file())

    def test_actual_step1_blocks_unconsented_sequence_and_invalid_fasta(self):
        with self.assertRaisesRegex(ValueError, "ALLOW_PUBLIC_MSA"):
            self.run_preflight("Paste sequence", text=QUERY)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.run_preflight("Upload FASTA", upload={"two.fasta": b">one\nACD\n>two\nEFG"}, consent=True)
        with self.assertRaisesRegex(ValueError, "No A3M selected"):
            self.run_preflight("Upload A3M")

    def test_actual_step1_changes_msa_and_clears_stale_prior_errors(self):
        body = next(node.body for node in ast.parse(self.setup).body if isinstance(node, ast.Try))
        selected, collecting = [], False
        for node in body:
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and node.value.args and isinstance(node.value.args[0], ast.Constant)
                    and node.value.args[0].value == "Preparing the input alignment"):
                collecting = True
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODEL_WEIGHT_SETTINGS" for t in node.targets):
                break
            if collecting:
                selected.append(node)
        state = {"msa": "old.a3m", "pdb": "old.pdb", "errors": {"msa": "pending", "pdb": "old error"}}
        generate = Mock(return_value=Path("new.a3m"))
        context = {"Path": Path, "ROOT": ROOT, "_selected_a3m": "", "_input_query": QUERY,
                   "input_state": state, "ALLOW_PUBLIC_MSA": True, "_cell_status": Mock(),
                   "im": SimpleNamespace(ensure_msa=generate),
                   "h": SimpleNamespace(validate_a3m=Mock(return_value={"query": QUERY, "records": 3}))}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "step1-publish-input", "exec"), context)
        self.assertEqual(state["msa"], "new.a3m")
        self.assertEqual(state["pdb"], "")
        self.assertEqual(state["errors"], {})
        self.assertEqual(state["msa_source"], "ColabFold MMseqs2")
        generate.assert_called_once()

    def test_canonical_client_is_embedded_verbatim(self):
        source = self.source_value(self.setup, "INPUT_SOURCE")
        self.assertEqual(source, MODULE_PATH.read_text(encoding="utf-8"))
        compile(source, "INPUT_SOURCE", "exec")

    def test_step1_contains_all_input_modes_and_explicit_privacy_consent(self):
        self.assertEqual(self.source_value(self.setup, "INPUT_MODE"), "Paste sequence")
        self.assertIs(self.source_value(self.setup, "ALLOW_PUBLIC_MSA"), False)
        for label in ("Upload FASTA", "Upload A3M", "public ColabFold", "Confidential", "confidential"):
            self.assertTrue(label.lower() in self.setup.lower() or label.lower() in "".join(self.notebook["cells"][-1]["source"]).lower())
        self.assertTrue(self.setup.index("_input_query=im.normalize_sequence") < self.setup.index("run(['nvidia-smi'"))
        self.assertTrue(self.setup.index("_msa_check=h.validate_a3m") < self.setup.index("MODEL_WEIGHT_SETTINGS=dict"))

    def test_generating_does_not_require_optional_msa_download_cell(self):
        definitions = [node.name for node in ast.walk(ast.parse(self.setup)) if isinstance(node, ast.FunctionDef)]
        self.assertIn("save_one_upload", definitions)
        self.assertIn("read_input_settings", definitions)
        optional = "".join(self.notebook["cells"][4]["source"])
        self.assertTrue(optional.startswith("# @title Step 2a · Download MSA (optional)"))
        self.assertNotIn("files.upload", optional)

    def test_explore_defaults_to_residue_and_restores_choice_after_2d(self):
        cell = next("".join(c["source"]) for c in self.notebook["cells"]
                    if c["cell_type"] == "code" and "VIEWER_SOURCE =" in "".join(c["source"]))
        source = self.source_value(cell, "VIEWER_SOURCE")
        self.assertIn('<option value="residue" selected>', source)
        self.assertIn("preferredColor = 'residue'", source)
        self.assertIn("manualFallback ? 'branch' : preferredColor", source)
        self.assertIn("colorMode.value === 'residue' ? 'spectrum'", source)
        compile(source, "VIEWER_SOURCE", "exec")

    def test_result_zip_still_includes_prepared_msa_without_weights(self):
        export = "".join(self.notebook["cells"][13]["source"])
        self.assertIn("extras.append((Path(MSA_PATH),'inputs/'", export)
        self.assertIn("No weights, environment or unrelated Drive data", export)


if __name__ == "__main__":
    unittest.main()
