"""Black-box optimization: find (x, y) minimizing a hidden quadratic bowl.

The model never sees the formula — only the loss at each point it probes. It must map
the surface from that feedback and locate the single minimum. The catch is in how the
bowl is built: it is randomized per run so the model cannot memorize or guess it, and
it is *anisotropic* — different curvatures on the two axes (a != b). That kills the
shortcut where the model assumes a symmetric paraboloid and solves it analytically from
a couple of points: with a != b, samples taken along a single line (e.g. y = x) are not
enough to identify the optimum, so the model has to probe off-axis and actually do the
black-box search, not recall a textbook formula.

The gate is deterministic and trusted, like every gate in this project: it reports the
*exact* loss and accepts a point once that loss falls below a small tolerance. The
difficulty lives in the hidden function, never in a noisy signal — the feedback the
model gets back is exact, just as subset-sum returns an exact "różnica".
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from agentic_loop import Candidate, History, Problem

_LIST_RE = re.compile(r"\[[^\]]*\]")
_TOLERANCE = 0.1  # success when the exact loss drops below this (the minimum value is 0)


@dataclass(frozen=True)
class Optimization(Problem):
    """One instance: a hidden anisotropic bowl with a single minimum (value 0)."""

    target_x: float
    target_y: float
    a: float  # curvature along x
    b: float  # curvature along y (a != b -> anisotropic, no symmetry shortcut)

    def loss(self, x: float, y: float) -> float:
        return self.a * (x - self.target_x) ** 2 + self.b * (y - self.target_y) ** 2

    def render_prompt(self, history: History) -> str:
        tries = "\n".join(f"[{c[0]}, {c[1]}] -> {fb}" for c, fb in history) or "(brak)"
        return (
            "Szukasz punktu (x, y) minimalizującego ukrytą, gładką funkcję (jeden minimum).\n"
            "Nie znasz wzoru — dostajesz tylko loss (im bliżej 0, tym lepiej). Funkcja nie "
            "jest symetryczna: próbki z jednej prostej mogą nie wystarczyć.\n"
            f"Poprzednie próby i loss:\n{tries}\n\n"
            "Na podstawie trendu loss zaproponuj lepszy punkt. Zwróć WYŁĄCZNIE listę JSON "
            "[x, y], np. [1.5, -0.5]. Bez komentarza."
        )

    def parse_answer(self, answer: str) -> list[Candidate]:
        points: list[Candidate] = []
        for match in _LIST_RE.findall(answer):
            try:
                nums = [float(v) for v in json.loads(match)]
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            if len(nums) == 2:
                points.append(nums)
        return points


def verify(problem: Optimization, candidate: Candidate) -> tuple[bool, str]:
    """The external gate: exact loss, deterministic verdict, no noise."""
    if not (
        isinstance(candidate, list)
        and len(candidate) == 2
        and all(isinstance(v, (int, float)) for v in candidate)
    ):
        return False, "kandydat musi być punktem [x, y]"
    x, y = candidate
    loss = problem.loss(x, y)
    if loss <= _TOLERANCE:
        return True, f"loss {loss:.4f} — w celu"
    return False, f"loss {loss:.4f} (minimalizuj do ~0)"


def generate(n: int = 12, seed: int | None = None, span: float = 5.0) -> Optimization:
    """Random hidden bowl so the model can't memorize or guess a round optimum.

    The optimum is kept to two decimals (non-round, e.g. 3.74, not 4.8) and the two
    curvatures differ, so the model must probe off-axis. `n` is unused — optimization
    has no instance size.
    """
    rng = random.Random(seed)
    return Optimization(
        target_x=round(rng.uniform(-span, span), 2),
        target_y=round(rng.uniform(-span, span), 2),
        a=round(rng.uniform(0.5, 3.0), 2),
        b=round(rng.uniform(0.5, 3.0), 2),
    )


def describe(problem: Optimization) -> dict[str, object]:
    """Full scale of the hidden bowl, for the log — the model never sees any of this, only
    the loss it gets back. Lets the analyst see the true optimum and how hard the instance is."""
    lo, hi = sorted((problem.a, problem.b))
    return {
        "kind": "optimization",
        "fully_observable": False,
        "optimum": [problem.target_x, problem.target_y],
        "a": problem.a,
        "b": problem.b,
        "anisotropy": round(hi / lo, 2),
        "tolerance": _TOLERANCE,
    }
