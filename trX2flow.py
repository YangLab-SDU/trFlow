"""Allow ``python -m trX2flow`` to launch the trX2-Flow CLI."""

from trx2flow.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
