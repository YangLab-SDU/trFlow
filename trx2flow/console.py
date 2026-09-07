"""Small dependency-free console presentation helpers."""
from __future__ import annotations

import os
import sys
import textwrap
from typing import Iterable, Tuple


LOGO = r'''
    .            ooooooo  ooooo   .oooo.           oooooooooooo oooo
  .o8             `8888    d8'  .dP""Y88b          `888'     `8 `888
.o888oo oooo d8b    Y888..8P          ]8P'          888          888   .ooooo.  oooo oooo    ooo
  888   `888""8P     `8888'         .d8P'           888oooo8     888  d88' `88b  `88. `88.  .8'
  888    888        .8PY888.      .dP'     8888888  888    "     888  888   888   `88..]88..8'
  888 .  888       d8'  `888b   .oP     .o          888          888  888   888    `888'`888'
  "888" d888b    o888o  o88888o 8888888888         o888o        o888o `Y8bod8P'     `8'  `8'
'''.strip("\n")

_WIDTH = 104


def _color(code: str, value: object) -> str:
    enabled = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    return f"\033[{code}m{value}\033[0m" if enabled else str(value)


def print_logo() -> None:
    print(_color("1;36", LOGO))
    print(_color("2", "                             Protein conformation sampling with flow matching"))


def panel(title: str, rows: Iterable[Tuple[str, object]]) -> None:
    """Print a compact, aligned key/value panel."""
    title_text = f" {title} "
    print("\n+" + title_text + "-" * max(1, _WIDTH - len(title_text) - 1) + "+")
    for key, value in rows:
        prefix = f"  {key:<16} "
        continuation = " " * len(prefix)
        value_lines = textwrap.wrap(
            str(value), width=_WIDTH - len(prefix), break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        for index, value_line in enumerate(value_lines):
            text = (prefix if index == 0 else continuation) + value_line
            print("|" + text.ljust(_WIDTH) + "|")
    print("+" + "-" * _WIDTH + "+")


def stage_start(sample: str, number: int, title: str) -> None:
    print(f"\n{_color('1;36', f'[{number}/3]')} {_color('1', sample)} :: {title}")


def stage_done(detail_text: str = "") -> None:
    suffix = f"  {detail_text}" if detail_text else ""
    print(f"      {_color('1;32', 'OK')}{suffix}")


def stage_skip(reason: str) -> None:
    print(f"      {_color('1;33', 'SKIP')}  {reason}")


def detail(message: str) -> None:
    print(f"      {_color('2', '>')} {message}")


def warning(message: str) -> None:
    print(f"\n{_color('1;33', 'WARNING')}  {message}")


def success(message: str) -> None:
    print(f"\n{_color('1;32', 'SUCCESS')}  {message}")
