"""Shared observability primitives for streamed makers.

A maker (Gemini, Ollama, ...) emits its prompt, streamed tokens, and final token usage to
these hooks. They are backend- and display-agnostic: a maker never imports a console or a
color, it just calls the optional callbacks. Token counts are model-specific data, so they
ride on the maker's hooks here — never on the generic `Maker` contract or `loop()`. Raw
counts only; cost is a separate offline concern (prices change).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .contracts import Candidate, History, Problem

Kind = Literal["thinking", "answer"]


@dataclass(frozen=True)
class Usage:
    """Token counts for one generation."""

    prompt_tokens: int
    candidate_tokens: int
    thoughts_tokens: int
    total_tokens: int
    model: str


@dataclass(frozen=True)
class StreamHooks:
    """Opt-in observability for a streamed generation. All hooks optional.

    on_prompt(text)      - the exact prompt sent this turn (input)
    on_token(kind, text) - each streamed chunk, kind in {"thinking", "answer"}
    on_usage(usage)      - token counts for the turn, once the stream completes
    """

    on_prompt: Callable[[str], None] | None = None
    on_token: Callable[[Kind, str], None] | None = None
    on_usage: Callable[[Usage], None] | None = None


class BaseStreamingMaker:
    """Shared scaffold for streaming LLM makers (Template Method).

    The shape every backend repeats — render the problem's prompt, fire `on_prompt`,
    stream the reply, hand the raw text to `problem.parse_answer` — lives here once; a
    backend only implements `_stream`. This is an implementation convenience, not the
    contract: the loop depends on the `Maker` Protocol (structural), so a maker need not
    inherit this — Gemini and Ollama just happen to share the shape.
    """

    hooks: StreamHooks  # set by each subclass's __init__

    def __call__(self, problem: Problem, history: History) -> list[Candidate]:
        prompt = problem.render_prompt(history)
        if self.hooks.on_prompt:
            self.hooks.on_prompt(prompt)
        return problem.parse_answer(self._stream(prompt))

    def _stream(self, prompt: str) -> str:
        """Send `prompt`, emit tokens/usage to `self.hooks`, return the visible answer."""
        raise NotImplementedError
