"""Install a wheel into a temporary target and smoke-test it outside the repo."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STATIC_MIME = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}


def command(arguments, cwd, environment, timeout=60):
    result = subprocess.run(arguments, cwd=cwd, env=environment, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode:
        raise AssertionError(f"Command failed ({result.returncode}): {arguments}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "wheel-results.json")
    args = parser.parse_args()
    report = {"passed": False, "checks": [], "platform": sys.platform}
    try:
        wheels = list(args.wheel_dir.resolve().glob("trflow-*.whl"))
        assert len(wheels) == 1, "Build into a fresh wheel directory containing exactly one trflow wheel."
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            static = sorted(name for name in names if name.startswith("trflow/web_static/") and not name.endswith("/"))
            expected = {"trflow/web_static/" + item.relative_to(ROOT / "trflow" / "web_static").as_posix()
                        for item in (ROOT / "trflow" / "web_static").rglob("*") if item.is_file()}
            assert expected <= names, "Wheel must include every published static asset and vendor notice."
            example = "trflow/web_assets/example/2akl.a3m"
            assert archive.read(example) == (ROOT / "example" / "msa" / "2akl.a3m").read_bytes()
            metadata = next(name for name in names if name.endswith(".dist-info/METADATA"))
            assert "Requires-Python: <3.12,>=3.11" in archive.read(metadata).decode("utf-8")
            report["static_assets"] = len(static)
            report["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
            report["checks"].append("wheel includes static assets, notices, exact example MSA, and supported Python metadata")
        with tempfile.TemporaryDirectory(prefix="trflow wheel α 测试 ") as folder:
            # Match canonical module/CWD paths even when Windows TEMP is aliased.
            root = Path(folder).resolve()
            installed = root / "installed library"
            working = root / "outside checkout 工作目录"
            working.mkdir()
            environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
            command([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile",
                     "--target", str(installed), str(wheel)], working, environment)
            environment["PYTHONPATH"] = str(installed)
            proof = command([sys.executable, "-c",
                "import json,trflow; from pathlib import Path; from importlib.metadata import distribution; "
                "from trflow.web import default_data_dir; "
                "d=distribution('trflow'); e=next(x for x in d.entry_points if x.name=='trFlow'); "
                "assert e.value=='trflow.cli:main'; "
                "print(json.dumps({'module':str(Path(trflow.__file__).resolve()),'data':str(default_data_dir().resolve())}))"],
                working, environment)
            proof = json.loads(proof)
            assert Path(proof["module"]).is_relative_to(installed.resolve()), "Do not accidentally import an editable checkout."
            assert Path(proof["data"]) == working / "outputs" / "web", "Installed defaults must not write into site-packages."
            for module in ("trFlow", "trflow.web", "run_web"):
                args_for_help = [sys.executable, "-m", module] + (["web"] if module == "trFlow" else []) + ["--help"]
                assert "--no-browser" in command(args_for_help, working, environment)
            console_help = command([sys.executable, "-c",
                "import sys; from importlib.metadata import distribution; sys.argv=['trFlow','web','--help']; "
                "next(e for e in distribution('trflow').entry_points if e.name=='trFlow').load()()"], working, environment)
            assert "--data-dir" in console_help
            report["checks"].append("isolated installed import, console entrypoint, module help commands, and CWD data directory")
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            log = root / "server.log"
            with log.open("w", encoding="utf-8") as stream:
                server = subprocess.Popen([sys.executable, "-m", "trFlow", "web", "--port", str(port),
                    "--no-browser", "--no-import-existing"], cwd=working, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 30
                    while True:
                        if server.poll() is not None:
                            raise AssertionError("Installed server exited: " + log.read_text(encoding="utf-8"))
                        try:
                            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                            connection.request("GET", "/api/targets")
                            response = connection.getresponse()
                            assert response.status == 200
                            assert json.loads(response.read())["targets"] == []
                            connection.close()
                            break
                        except (ConnectionError, OSError):
                            if time.monotonic() > deadline:
                                raise TimeoutError("Installed server startup exceeded 30 seconds.")
                            time.sleep(0.1)
                    for route in ["/", "/api/config", "/api/example"] + ["/static/" + name.removeprefix("trflow/web_static/") for name in static]:
                        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                        try:
                            connection.request("GET", route)
                            response = connection.getresponse()
                            assert response.status == 200, route
                            content = response.read()
                            assert content, route
                            suffix = Path(route).suffix if route != "/" else ".html"
                            if suffix in STATIC_MIME:
                                assert response.getheader("Content-Type").split(";")[0] == STATIC_MIME[suffix], route
                            if route == "/api/example":
                                sample = json.loads(content)
                                assert sample["name"] == "2akl" and sample["msa_filename"] == "2akl.a3m"
                                assert sample["msa_text"] == (ROOT / "example" / "msa" / "2akl.a3m").read_text(encoding="utf-8")
                        finally:
                            connection.close()
                    assert (working / "outputs" / "web" / "targets.sqlite").is_file()
                    assert not (installed / "outputs").exists()
                    report["checks"].append("actual installed loopback server serves config, empty queue, packaged example, and all static assets with stable MIME")
                finally:
                    if server.poll() is None:
                        server.terminate()
                        try:
                            server.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            server.kill()
                            server.wait(timeout=10)
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
        print(str(exc), file=sys.stderr)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
