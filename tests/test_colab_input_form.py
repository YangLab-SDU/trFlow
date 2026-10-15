"""Conditional input-card tests; optional loopback UI fixture, no MSA requests."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "colab" / "input_form.py"
spec = importlib.util.spec_from_file_location("colab_form_under_test", MODULE_PATH)
form = importlib.util.module_from_spec(spec)
spec.loader.exec_module(form)


class InputFormTests(unittest.TestCase):
    def settings(self, **changes):
        value = dict(form.DEFAULT_SETTINGS, allow_public=True, sequence="ACDEFGHIK")
        value.update(changes)
        return value

    def test_default_card_has_sequence_and_collapsed_msa_options(self):
        markup, script = form.input_form(token="trflow_input_test")
        self.assertIn('<div data-role="sequence-area">', markup)
        self.assertIn('aria-label="SEQUENCE"', markup)
        self.assertNotIn("QUERY_SEQUENCE", markup)
        self.assertIn('<details data-role="advanced">', markup)
        self.assertNotRegex(markup, r'<details[^>]*\bopen\b')
        self.assertIn('<input data-role="consent" type="checkbox">', markup)
        self.assertIn('value="env" selected', markup)
        self.assertIn('value="20" selected', markup)
        self.assertIn('type="checkbox" checked', markup)
        self.assertIn("sequence.disabled = !paste", script)
        self.assertIn("sequence.required = paste", script)

    def test_upload_modes_hide_sequence_before_javascript_runs(self):
        for mode in ("Upload FASTA", "Upload A3M"):
            with self.subTest(mode=mode):
                markup, _ = form.input_form({"mode": mode})
                self.assertIn('<div data-role="sequence-area" hidden>', markup)
                self.assertIn('data-role="upload-hint" class="tf-hint">', markup)
                if mode == "Upload A3M":
                    self.assertIn('<div data-role="msa-area" hidden>', markup)

    def test_rerun_requires_public_consent_again(self):
        markup, _ = form.input_form(self.settings())
        self.assertIn('<input data-role="consent" type="checkbox">', markup)

    def test_form_mounts_in_current_output_frame_without_external_scripts(self):
        markup, script = form.input_form()
        self.assertIn("document.body.appendChild(mount)", script)
        self.assertIn("new Promise(resolve", script)
        self.assertNotIn("<script", markup.lower())
        self.assertNotIn("fetch(", script)
        self.assertNotIn("invokeFunction", script)
        self.assertNotIn("MARKUP_JSON", script)
        self.assertNotIn("TOKEN_JSON", script)

    def test_sequence_prefill_is_escaped_and_token_is_checked(self):
        markup, _ = form.input_form({"sequence": '</textarea><script>alert("x")</script>'})
        self.assertIn("&lt;/textarea&gt;&lt;script&gt;", markup)
        self.assertNotIn('<script>alert', markup)
        with self.assertRaises(ValueError):
            form.input_form(token='trflow_input_x" onclick="bad')

    def test_unique_ids_do_not_collide_between_reruns(self):
        first, _ = form.input_form()
        second, _ = form.input_form()
        self.assertNotEqual(re.search(r'id="([^"]+)"', first).group(1),
                            re.search(r'id="([^"]+)"', second).group(1))

    def test_pasted_sequence_requires_public_consent(self):
        self.assertEqual(form.validate_settings(self.settings())["sequence"], "ACDEFGHIK")
        with self.assertRaisesRegex(ValueError, "Confirm public MSA"):
            form.validate_settings(self.settings(allow_public=False))

    def test_hidden_sequence_and_public_flag_are_ignored_for_a3m_upload(self):
        selected = form.validate_settings(self.settings(mode="Upload A3M", allow_public=True))
        self.assertEqual(selected["sequence"], "")
        self.assertIs(selected["allow_public"], False)

    def test_fasta_clears_pasted_sequence_but_still_requires_consent(self):
        selected = form.validate_settings(self.settings(mode="Upload FASTA"))
        self.assertEqual(selected["sequence"], "")
        with self.assertRaises(ValueError):
            form.validate_settings(self.settings(mode="Upload FASTA", allow_public=False))

    def test_rejects_invalid_or_unbounded_payload_settings(self):
        for changes in ({"mode": "unexpected"}, {"use_env": 1}, {"use_filter": "false"},
                        {"allow_public": 1}, {"sequence": []}, {"timeout_minutes": True},
                        {"timeout_minutes": 0}, {"timeout_minutes": 120}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                form.validate_settings(self.settings(**changes))
        with self.assertRaises(ValueError):
            form.validate_settings(None)

    def test_notebook_embeds_reviewed_form_source_and_waits_for_user(self):
        notebook = json.loads((ROOT / "trFlow.ipynb").read_text(encoding="utf-8"))
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if cell.get("metadata", {}).get("id") == "trflow-input-prepare")
        assignment = next(node for node in ast.walk(ast.parse(source))
                          if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                          and t.id == "INPUT_FORM_SOURCE" for t in node.targets))
        self.assertEqual(ast.literal_eval(assignment.value), MODULE_PATH.read_text(encoding="utf-8"))
        self.assertIn("output.eval_js(_input_javascript,timeout_sec=None)", source)
        self.assertIn("Run this cell", source)
        self.assertIn("file picker", source)


def fixture_html():
    _, script = form.input_form(token="trflow_input_browser_test")
    # Exercise the same mounting path as eval_js, with no pre-rendered form.
    return ('<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>trFlow Step 2 input QA</title></head><body style="margin:20px">'
            '<h1>Step 2 input panel · synthetic QA</h1><p>No uploads or MSA searches are sent by this fixture.</p>'
            '<script>' + script + '.then(value => { const report = document.createElement("pre");'
            'report.id="accepted-input"; report.textContent=JSON.stringify(value,null,2);'
            'document.body.appendChild(report); });</script></body></html>')


def serve_fixture():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import os

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            data = fixture_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print("INPUT_FORM_FIXTURE http://127.0.0.1:" + str(server.server_address[1]) + "/ PID " + str(os.getpid()), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--serve-fixture"]:
        serve_fixture()
    else:
        unittest.main()
