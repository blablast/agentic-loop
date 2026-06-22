"""Console styling and live streaming output — dependency-free on purpose.

One module owns every ANSI escape code, so nothing else in the codebase hardcodes
a color. `Console` handles *how* to paint; `StreamView` handles *what* the token
stream looks like. Neither the maker nor the loop knows about either — they talk to
plain callables, so display stays swappable.
"""

from __future__ import annotations

import os
import sys

# Single source of truth for styling.
_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "gray": "\033[90m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "red": "\033[31m",
}


class Console:
    """Writes optionally-colored text to stdout. Colors auto-disable when the
    output is not a TTY or when NO_COLOR is set, so logs/pipes stay clean."""

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = (
            enabled
            if enabled is not None
            else (sys.stdout.isatty() and "NO_COLOR" not in os.environ)
        )

    def paint(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles:
            return text
        codes = "".join(_CODES[s] for s in styles)
        return f"{codes}{text}{_CODES['reset']}"

    def write(self, text: str, *styles: str) -> None:
        sys.stdout.write(self.paint(text, *styles))
        sys.stdout.flush()

    def line(self, text: str = "", *styles: str) -> None:
        self.write(text + "\n", *styles)


class StreamView:
    """Renders a stream of (kind, text) tokens, coloring each kind and printing a
    section header the first time it appears. Single responsibility: turn a token
    stream into readable, colored console output."""

    # kind -> (header label, *styles)
    SECTIONS = {
        "prompt": ("prompt", "blue"),
        "thinking": ("thinking", "gray"),
        "answer": ("answer", "cyan"),
    }

    def __init__(self, console: Console) -> None:
        self.console = console
        self._current: str | None = None

    def token(self, kind: str, text: str) -> None:
        """Feed one streamed token. Switches section header when `kind` changes."""
        if kind != self._current:
            self.end()
            self._open(kind)
        self.console.write(text, *self._styles(kind))

    def block(self, kind: str, text: str) -> None:
        """Render a complete, non-streamed section (e.g. the prompt sent that turn)."""
        self.end()
        self._open(kind)
        self.console.line(text, *self._styles(kind))
        self._current = None

    def end(self) -> None:
        """Close the current streamed section."""
        if self._current is not None:
            self.console.line()
            self._current = None

    def _open(self, kind: str) -> None:
        label = self.SECTIONS[kind][0]
        self.console.line(f"  ┌─ {label}", "bold", "dim")
        self._current = kind

    def _styles(self, kind: str) -> tuple[str, ...]:
        return self.SECTIONS[kind][1:]
