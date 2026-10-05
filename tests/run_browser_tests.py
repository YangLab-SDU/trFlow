"""Run mock/fixture browser tests without contacting an existing workbench."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
PORT = 8766  # web_ui_smoke refuses all other ports before any mutation.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "browser")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    node = os.environ.get("NODE_EXECUTABLE") or shutil.which("node")
    if not node:
        parser.error("Node.js >=20 is required; install dev dependencies with npm ci.")
    # Refuse an occupied port instead of attaching to, restarting, or killing
    # somebody else's server. The fixture owns a fresh temporary data directory.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", PORT))
        except OSError:
            parser.error("Port 8766 is occupied. Stop your own QA fixture before retrying.")
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    report = {"passed": False, "fixture_port": PORT, "steps": [], "real_workbench_requests": 0}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="trflow-browser-fixture-") as folder:
        log_path = output / "fixture-server.log"
        with log_path.open("w", encoding="utf-8") as log:
            server = subprocess.Popen(
                [sys.executable, "-B", str(ROOT / "tests" / "test_web_delete.py"),
                 "--serve-fixture", "--port", str(PORT), "--data-dir", folder],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 45
                base = f"http://127.0.0.1:{PORT}"
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("QA fixture exited before startup; inspect fixture-server.log.")
                    try:
                        with urllib.request.urlopen(base + "/api/targets", timeout=2) as response:
                            targets = json.load(response)["targets"]
                        if len(targets) == 3 and all(item["name"].startswith("QA ") for item in targets):
                            break
                        raise RuntimeError("Refusing a service that does not contain exactly three QA fixtures.")
                    except (urllib.error.URLError, TimeoutError, ConnectionError):
                        if time.monotonic() > deadline:
                            raise TimeoutError("QA fixture did not become ready within 45 seconds.")
                        time.sleep(0.1)
                for script in ("web_sampling_modes.cjs", "web_i18n_errors.cjs", "web_ui_smoke.cjs"):
                    destination = output / Path(script).stem
                    result = subprocess.run(
                        [node, str(ROOT / "tests" / script), base, str(destination)],
                        cwd=ROOT, env=environment, text=True, encoding="utf-8", errors="replace",
                        capture_output=True, timeout=180,
                    )
                    (output / (Path(script).stem + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
                    print(result.stdout, end="")
                    if result.stderr:
                        print(result.stderr, file=sys.stderr, end="")
                    report["steps"].append({"script": script, "returncode": result.returncode})
                    if result.returncode:
                        raise RuntimeError(f"{script} failed; inspect its report and log.")
                report["passed"] = True
            except Exception as exc:
                report["error"] = str(exc)
                print(str(exc), file=sys.stderr)
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=10)
    report["duration_seconds"] = round(time.monotonic() - started, 3)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
