"""Concrete problems for the agentic loop — a layer on top of the `agentic_loop` core.

Each module here is a self-contained case: a `Problem` subclass, its external gate
(`verify`), an instance generator, and any problem-specific makers. The dependency
points one way only — `problems` imports `agentic_loop`, never the reverse.

Problems are listed in `REGISTRY` so a runner can pick one by name. Adding a problem =
add its module and one `REGISTRY` line; nothing in the core or the runners changes.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentic_loop import Problem, Verifier

from . import maze, optimization, regex_golf, reverse_engineering, subset_sum

Describer = Callable[[Any], dict[str, object]]
"""`describe(instance) -> dict`: the full scale of an instance (including parts hidden
from the model), recorded in the log so an analyst can read the problem's true size."""


@dataclass(frozen=True)
class ProblemSpec:
    """Everything a generic (LLM-maker) runner needs to drive a problem by name.

    `generate(n, seed)` builds an instance; `verify` is its external gate; `describe`
    reports the instance's full scale for the log. A runner pairs these with the generic
    `GeminiMaker` — no problem-specific code needed.
    """

    name: str
    generate: Callable[[int, int | None], Problem]
    verify: Verifier
    describe: Describer


REGISTRY: dict[str, ProblemSpec] = {
    "subset_sum": ProblemSpec(
        "subset_sum", subset_sum.generate, subset_sum.verify, subset_sum.describe
    ),
    "optimization": ProblemSpec(
        "optimization", optimization.generate, optimization.verify, optimization.describe
    ),
    "regex_golf": ProblemSpec(
        "regex_golf", regex_golf.generate, regex_golf.verify, regex_golf.describe
    ),
    "reverse_engineering": ProblemSpec(
        "reverse_engineering",
        reverse_engineering.generate,
        reverse_engineering.verify,
        reverse_engineering.describe,
    ),
    "maze": ProblemSpec("maze", maze.generate, maze.verify, maze.describe),
}


def get(name: str) -> ProblemSpec:
    """Look up a registered problem, or raise with the list of known names."""
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown problem {name!r}; known: {known}") from None
