"""trFlow entry point.

Bootstrap sys.path so the vendored ``esm`` package (absolute ``import esm``)
and the copied OpenFold source (``openfold/``) resolve correctly, then delegate
to the trflow package CLI.

The installed ``trFlow predict`` command is the primary interface. This
script remains as a source-tree-compatible alternative.

Usage:
    python run_trflow.py predict INPUT [options]
    python run_trflow.py evaluate --pred-dir PREDICTIONS --native-dir REFERENCES
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

# The vendored ESM package uses absolute imports (`import esm`), so the repo
# root (where esm/ and openfold/ live) must be on sys.path.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "openfold"))

from trflow.cli import main

if __name__ == "__main__":
    main()
