"""The agentic loop: a maker proposes, an external verifier decides.

Materializes the four elements of the pattern:
  STATE   - `history`, the memory of attempts and their feedback
  EXECUTE - `maker(problem, history)` proposes a batch of candidates
  VERIFY  - `verify(problem, candidate)` checks each (the external gate, never the maker)
  ITERATE - append each failure to history and try again
  STOP    - success, or a hard iteration limit

Because verifying is cheap, a maker may return several candidates per turn and the loop
checks them all — the first that passes wins (an LLM often emits more than one list in a
single answer). One iteration = one maker call (the expensive resource), not one candidate.

The loop is problem-agnostic: it depends only on the `Problem` / `Maker` / `Verifier`
contracts, so any problem and any maker plug into the identical control flow. Display
is injected via `LoopView` callbacks — the loop knows nothing about printing or color;
with no view it runs silently.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .contracts import Candidate, History, Maker, Problem, Verifier


@dataclass(frozen=True)
class LoopView:
    """Optional, immutable display hooks. Omit to run the loop silently.

    on_attempt(i)                    - before attempt i (e.g. print a header)
    on_verdict(i, candidate, ok, fb) - after the gate decides
    """

    on_attempt: Callable[[int], None] | None = None
    on_verdict: Callable[[int, Candidate, bool, str], None] | None = None


def loop(
    problem: Problem,
    maker: Maker,
    verify: Verifier,
    *,
    max_iters: int = 40,
    view: LoopView | None = None,
) -> tuple[Candidate | None, int]:
    """Iterate until the gate accepts a candidate or `max_iters` is hit."""
    view = view or LoopView()
    history: History = []  # STATE
    for i in range(1, max_iters + 1):  # STOP: hard limit (one maker call each)
        if view.on_attempt:
            view.on_attempt(i)
        for candidate in maker(problem, history):  # EXECUTE: maker proposes a batch
            ok, feedback = verify(problem, candidate)  # VERIFY: cheap gate, run on each
            if view.on_verdict:
                view.on_verdict(i, candidate, ok, feedback)
            if ok:
                return candidate, i  # STOP: success
            history.append((candidate, feedback))  # ITERATE: remember the failure
    return None, max_iters
