# Development and CPU regression tests

The automated suite supports the project's declared Python range: **Python 3.11**.
GitHub Actions runs the same tests on `ubuntu-latest` and `windows-latest`, using
PyTorch 2.6.0 CPU wheels. These checks do not download model weights, run OpenFold,
run a real structure-model prediction, or validate GPU/CUDA inference. The real
sampling loop is covered with small CPU tensors and a recording model double.

## Python setup

Create and activate a Python 3.11 virtual environment, then run the following in
the repository root on either platform:

```text
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

`requirements-test.txt` is a **test environment**, not a full prediction install.
It keeps NumPy below 2, uses the project's Biopython 1.80, and installs the pinned
dm-tree 0.1.9 binary wheel needed by the CPU core imports. Python 3.11 x86-64 wheels
exist for both Windows and Linux. Do not install the CUDA-oriented
`requirements.txt` merely to run these tests. GPU prediction setup is documented
in the main README instead.

For a JSON report with package versions and platform information:

```text
python tests/run_cpu_tests.py --report artifacts/cpu-results.json
```

Web tests use temporary directories, ephemeral loopback ports, a disabled
prediction worker, or explicitly mocked subprocesses. They exercise queue
persistence, request validation, cancellation/deletion/restore, path safety,
portable filenames, error codes/translations, CPU alignment/clustering, and ZIP
contents without changing an existing workbench. The native analysis suite uses
a fresh subprocess to avoid mixing platform-specific numerical-library runtimes.

## Build and installed-package checks

```text
python -m build
python tests/verify_wheel.py --wheel-dir dist --report artifacts/wheel-results.json
```

Use a fresh `dist` directory with exactly one trFlow wheel. The verifier installs
that wheel with `--no-index --no-deps` into a temporary target, imports it from a
separate working directory containing spaces and non-ASCII characters, checks
console/module `--help` entrypoints, and starts a disposable loopback server. It
verifies bundled JavaScript/CSS/SVG/vendor notices, the packaged example MSA,
stable MIME types, an empty queue, and that default state goes under the working
directory rather than installed package files. No prediction is submitted.

## Browser tests

Node.js 20 or newer and the pinned development-only Playwright dependency are
required; CI uses Node.js 24. From the repository root:

```text
npm ci
npx playwright install chromium
python tests/run_browser_tests.py --output artifacts/browser
```

On Linux, use `npx playwright install --with-deps chromium` to install Chromium's
system libraries when needed. The runner reserves port **8766**, refuses an
occupied port, starts `test_web_delete.py --serve-fixture` with a new temporary
directory, and terminates only its own process afterward. It runs:

| Script | Scope | API safety |
| --- | --- | --- |
| `web_sampling_modes.cjs` | generation-mode exclusivity, payloads, retained uploads, bilingual/mobile layout | all API responses mocked |
| `web_i18n_errors.cjs` | structured/legacy error localization and preserved diagnostics | all API responses mocked |
| `web_ui_smoke.cjs` | viewer state, bilingual/mobile layout, delete/cancel/undo workflow | permits mutations only on the isolated port-8766 tiny QA fixtures |

The default browser is Playwright's bundled Chromium. Set `BROWSER_CHANNEL=chrome`
or `BROWSER_CHANNEL=msedge` to use an installed channel, or `CHROME_EXECUTABLE` for
an explicit browser executable. `PLAYWRIGHT_MODULE`, `NODE_EXECUTABLE`, and
`PYTHON_EXECUTABLE` are optional local development overrides, not CI requirements.

`web_smoke.cjs`, `web_render_smoke.cjs`, and `web_download_focus.cjs` are optional
read-only checks for a service containing completed ensembles. They block
non-GET browser API requests and do not submit or delete targets. The render
script specifically requires a completed 200-conformation ensemble, so it is not
part of the synthetic default CI suite. Keep genuine user outputs and private
logs out of committed test data and uploaded CI artifacts.

## CI results and limits

The workflow uses only `contents: read`, runs on ordinary `push`/`pull_request`
events, and never exposes repository secrets or uses `pull_request_target`.
Uploaded artifacts contain test-generated reports, logs, and screenshots from
mocked or disposable synthetic fixtures, with seven-day retention. A configured
Linux job is not evidence that Linux tests passed: check the actual GitHub Actions
run before claiming platform validation. The same applies to GPU inference,
which remains outside this CPU-only workflow.
