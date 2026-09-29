"""Backward-compatible entry point for trFlow structure evaluation.

The installed command is preferred::

    trFlow evaluate --pred-dir PREDICTIONS --native-dir REFERENCES

This script retains the original source-tree interface::

    python evaluate.py --pred-dir PREDICTIONS --native-dir REFERENCES
"""

from trflow.evaluation import main


if __name__ == "__main__":
    raise SystemExit(main())
