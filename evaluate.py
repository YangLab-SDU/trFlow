"""Backward-compatible entry point for trX2-Flow structure evaluation.

The installed command is preferred::

    trX2flow evaluate --pred-dir PREDICTIONS --native-dir REFERENCES

This script retains the original source-tree interface::

    python evaluate.py --pred-dir PREDICTIONS --native-dir REFERENCES
"""

from trx2flow.evaluation import main


if __name__ == "__main__":
    raise SystemExit(main())
