"""trX2-Flow entry point.

Bootstrap sys.path so the vendored ``esm`` package (absolute ``import esm``)
and the copied OpenFold source (``openfold/``) resolve correctly, then delegate
to the trx2flow package CLI.

The installed ``trX2flow predict`` command is the primary interface. This
script remains as a source-tree-compatible alternative.

Usage:
    python run_trx2flow.py predict INPUT [options]
    python run_trx2flow.py evaluate --pred-dir PREDICTIONS --native-dir REFERENCES
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

# The vendored ESM package uses absolute imports (`import esm`), so the repo
# root (where esm/ and openfold/ live) must be on sys.path.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "openfold"))

from trx2flow.cli import main

if __name__ == "__main__":
    main()
