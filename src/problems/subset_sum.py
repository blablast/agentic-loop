"""Subset-sum: find a subset of `numbers` summing exactly to `target`.

Easy to verify (linear), NP-complete to find — the generator–verifier asymmetry the
whole demo is about. This module is the *concrete* case: the problem object, its
external gate, an instance generator, and two subset-sum-specific makers. The generic
core (`loop`, `GeminiMaker`) knows nothing about any of it.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations

from agentic_loop import Candidate, History, Problem

_HALF = 2  # split the set roughly in half for the heuristic's first guess
_MIN_HIDDEN = 3  # smallest hidden subset the generator builds a target from
_LIST_RE = re.compile(r"\[[^\]]*\]")  # a JSON-looking list of ints in the model's answer


@dataclass(frozen=True)
class SubsetSum(Problem):
    """An instance of subset-sum. Immutable, so it is safe to share and hash."""

    numbers: tuple[int, ...]
    target: int

    def render_prompt(self, history: History) -> str:
        tries = "\n".join(f"{c} -> {fb}" for c, fb in history) or "(brak)"
        return (
            f"Zbiór S = {list(self.numbers)}.\n"
            f"Znajdź podzbiór S, który sumuje się dokładnie do {self.target}.\n"
            f"Poprzednie próby i feedback weryfikatora:\n{tries}\n\n"
            "Wykorzystaj feedback. Zwróć WYŁĄCZNIE listę JSON liczb, np. [3, 7, 12]. "
            "Bez komentarza, bez wyjaśnień."
        )

    def parse_answer(self, answer: str) -> list[Candidate]:
        """Pull every JSON int-list out of the model's answer (it often emits several)."""
        candidates: list[Candidate] = []
        for match in _LIST_RE.findall(answer):
            try:
                candidates.append([int(x) for x in json.loads(match)])
            except (ValueError, json.JSONDecodeError):
                continue
        return candidates


def verify(problem: SubsetSum, candidate: Candidate) -> tuple[bool, str]:
    """The external gate. Feedback is concrete so the next attempt can reuse it."""
    if not isinstance(candidate, (list, tuple)) or not all(isinstance(x, int) for x in candidate):
        return False, "kandydat musi być listą liczb całkowitych"
    if any(x not in problem.numbers for x in candidate):
        return False, "element spoza zbioru"
    if len(set(candidate)) != len(candidate):
        return False, "powtórzony element"  # a subset uses each number at most once
    total = sum(candidate)
    if total != problem.target:
        return False, f"różnica {problem.target - total:+d}"
    return True, "OK"


def generate(n: int = 12, seed: int | None = None, low: int = 2, high: int = 40) -> SubsetSum:
    """Return a random instance. The target sums a hidden subset, so a solution exists
    and the model cannot have memorized it."""
    rng = random.Random(seed)
    numbers = tuple(sorted(rng.sample(range(low, high + 1), n)))
    hidden = rng.sample(numbers, rng.randint(_MIN_HIDDEN, max(_MIN_HIDDEN, n // 2)))
    return SubsetSum(numbers=numbers, target=sum(hidden))


def describe(problem: SubsetSum) -> dict[str, object]:
    """Full scale of the instance, for the log. Subset-sum is fully observable — the model
    already sees `numbers` and `target` — so this just records the search-space size."""
    return {
        "kind": "subset_sum",
        "fully_observable": True,
        "n": len(problem.numbers),
        "numbers": list(problem.numbers),
        "target": problem.target,
        "search_space": 2 ** len(problem.numbers),
    }


class BruteForceMaker:
    """Deterministic — proves the loop works without a model or an API key.

    Enumerates subsets. Inefficient (exponential), which is exactly the point:
    generating is expensive, verifying is cheap.
    """

    def __init__(self) -> None:
        self._subsets: Iterator[Candidate] | None = None

    def __call__(self, problem: SubsetSum, history: History) -> list[Candidate]:
        if self._subsets is None:
            self._subsets = self._all_subsets(problem.numbers)
        return [next(self._subsets)]

    @staticmethod
    def _all_subsets(numbers: Sequence[int]) -> Iterator[Candidate]:
        for r in range(1, len(numbers) + 1):
            for combo in combinations(numbers, r):
                yield list(combo)


class HeuristicMaker:
    """Stands in for an LLM without an API key: guesses, then uses the gate's feedback
    about the difference to correct itself (Reflexion-style)."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def __call__(self, problem: SubsetSum, history: History) -> list[Candidate]:
        numbers = problem.numbers
        if not history:
            return [sorted(self.rng.sample(numbers, self._half(numbers)))]

        last, _ = history[-1]
        candidate = list(last)
        diff = problem.target - sum(candidate)

        if diff > 0:  # too small - add the element closest to the gap
            free = [x for x in numbers if x not in candidate]
            options = [x for x in free if x <= diff] or free
            if options:
                candidate.append(min(options, key=lambda x: abs(diff - x)))
        elif diff < 0 and candidate:  # too big - drop the element closest to the excess
            candidate.remove(min(candidate, key=lambda x: abs(x + diff)))
        else:  # diff 0 but an invalid element slipped in - reshuffle
            return [sorted(self.rng.sample(numbers, self._half(numbers)))]

        return [sorted(set(candidate))]

    @staticmethod
    def _half(numbers: Sequence[int]) -> int:
        return max(1, round(len(numbers) / _HALF))
