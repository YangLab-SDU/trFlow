"""Allow ``python -m trFlow`` to launch the trFlow CLI."""

from trflow.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
