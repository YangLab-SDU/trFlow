"""CPU-only notebook preview fixtures; optional disposable loopback UI check."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_viewer():
    notebook = json.loads((ROOT / "trFlow.ipynb").read_text(encoding="utf-8"))
    cell = next("".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code" and "VIEWER_SOURCE =" in "".join(cell["source"]))
    source = next(ast.literal_eval(node.value) for node in ast.walk(ast.parse(cell))
                  if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                  and target.id == "VIEWER_SOURCE" for target in node.targets))
    spec = importlib.util.spec_from_loader("trflow_colab_viewer_fixture", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, "embedded-notebook-viewer", "exec"), module.__dict__)
    return module


def synthetic_ensemble(count=12):
    """Generated geometry only; no real structure, model inference or user data."""
    conformers = []
    for index in range(count):
        atoms = []
        for residue in range(1, 31):
            angle = residue * 1.75
            x = 2.3 * math.cos(angle) + 0.025 * index * residue
            y = 2.3 * math.sin(angle)
            z = residue * 1.5 + 0.5 * math.sin(residue * 0.3 + index)
            tangent = (-math.sin(angle), math.cos(angle), 0.43)
            positions = {
                "N": (x - tangent[0], y - tangent[1], z - tangent[2]),
                "CA": (x, y, z),
                "C": (x + tangent[0], y + tangent[1], z + tangent[2]),
                "O": (x + tangent[0] + math.cos(angle),
                      y + tangent[1] + math.sin(angle), z + tangent[2]),
                "CB": (x + 1.4 * math.cos(angle), y + 1.4 * math.sin(angle), z),
            }
            for name, (ax, ay, az) in positions.items():
                serial = len(atoms) + 1
                atoms.append(f"ATOM  {serial:5d} {name:^4s} ALA A{residue:4d}    "
                             f"{ax:8.3f}{ay:8.3f}{az:8.3f}  1.00  0.00          {name[0]:>2s}  ")
        conformers.append({
            "pdb": "\n".join(atoms) + "\nEND\n",
            "metadata": {"file": f"synthetic_{index+1:03d}.pdb", "model": "xray" if index % 2 else "nmr",
                         "init_source": "synthetic", "ca_rmsd_angstrom": index * 0.1,
                         "common_ca": 30, "query_length": 30, "query_coverage": 1.0},
        })
    return SimpleNamespace(conformers=conformers, report={
        "reference": "synthetic_001.pdb", "total_predictions": count,
        "selection_seed": 42, "branches": ["xray/synthetic", "nmr/synthetic"],
    })


class ColabViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.viewer = load_viewer()

    def test_real_embedded_viewer_builds_html_with_residue_default(self):
        markup = self.viewer.ensemble_html(synthetic_ensemble())
        self.assertIn('<option value="residue" selected>Residue index (spectrum)</option>', markup)
        self.assertIn("Loaded 12 of 12 conformers", markup)
        self.assertIn("Residue position: N-terminus", markup)
        self.assertIn("Colors default to residue position", markup)
        self.assertNotIn("TOKEN_JSON", markup)
        self.assertNotIn("PAYLOAD", markup)

    def test_source_branch_option_and_static_fallback_remain_available(self):
        markup = self.viewer.ensemble_html(synthetic_ensemble(), overlay_limit=3)
        self.assertIn('<option value="branch">Source branch</option>', markup)
        self.assertIn("2D Cα projection", markup)
        self.assertIn("manualFallback ? 'branch' : preferredColor", markup)
        self.assertIn("preferredColor = 'residue'", markup)
        self.assertIn('"overlayLimit":3', markup.replace(" ", ""))

    def test_overlay_and_preview_bounds_are_unchanged(self):
        for count in (0, -1, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.viewer.ensemble_html(synthetic_ensemble(), overlay_limit=count)


def serve_fixture():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    viewer = load_viewer()
    markup = ('<!doctype html><html><head><meta charset="utf-8"><title>trFlow Colab preview QA</title>'
              '<meta name="viewport" content="width=device-width,initial-scale=1"></head>'
              '<body style="margin:20px"><h1>Colab Explore · synthetic QA fixture</h1>'
              '<p>Generated geometry only. No prediction or user files.</p>'
              + viewer.ensemble_html(synthetic_ensemble()) + '</body></html>').encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(markup)))
            self.end_headers()
            self.wfile.write(markup)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print("COLAB_VIEWER_FIXTURE http://127.0.0.1:" + str(server.server_address[1]) + "/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve-fixture"]:
        serve_fixture()
    else:
        unittest.main()
