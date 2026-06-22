"""Core contracts of the agentic loop — the abstractions everything depends on.

The generic harness (`loop`) and the generic LLM maker (`GeminiMaker`) depend only on
these types, never on a concrete problem. A new problem implements `Problem`; a new
solver satisfies `Maker`; both plug into the loop without changing it.

Why `verify` is a separate `Verifier`, not a method on the maker: the gate that decides
success must be external to the solver, because a model's self-assessment is unreliable.
That generator–verifier asymmetry is the whole point of the demo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol

Candidate = Any  # opaque to the core — each problem defines its own candidate shape
Attempt = tuple[Candidate, str]
History = list[Attempt]


class Problem(ABC):
    """A task the loop can pose and (via a `Verifier`) check.

    A concrete problem owns both ends of the conversation with an LLM maker: how to
    present the task (`render_prompt`) and how to read the model's raw answer back into
    candidates (`parse_answer`). A candidate's shape is problem-defined — for subset-sum
    it is a list of ints, but another problem might use a string, a number, a graph —
    so the core treats candidates as opaque.
    """

    @abstractmethod
    def render_prompt(self, history: History) -> str:
        """Build the prompt for an LLM maker, folding in the verifier's feedback so far."""

    @abstractmethod
    def parse_answer(self, answer: str) -> list[Candidate]:
        """Extract every candidate the model's raw answer contains. Verifying is cheap,
        so one answer may yield several candidates for the loop to check."""


class Maker(Protocol):
    """The "generate" half: propose candidates for `problem`, given past attempts.

    Returns a *list* of candidates, not one — verifying is cheap, so the loop checks
    every proposal a turn yields (an LLM often emits several lists in a single answer).
    A maker with one idea returns a one-element list.

    `problem` is typed `Any` on purpose — a concrete maker narrows it to the problem
    type it understands (e.g. subset-sum's makers read `problem.numbers`). The runner
    only ever pairs a problem with a maker that fits it.
    """

    def __call__(self, problem: Any, history: History) -> list[Candidate]: ...


Verifier = Callable[[Any, Candidate], tuple[bool, str]]
"""The external gate: `(problem, candidate) -> (ok, concrete feedback)`. Deliberately
not part of the maker — concrete feedback is fed back into the next attempt, which is
what makes the loop agentic rather than blind retry."""
