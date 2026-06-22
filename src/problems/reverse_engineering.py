"""Reverse engineering: deduce a hidden input->output rule from feedback.

The model sees a few solved examples and must predict the outputs for a set of query
inputs. The gate is the only place the rule lives; on a miss it reports which queries are
wrong and the model's *own* wrong output — never the expected one — so the model has to
infer the rule rather than parrot it.

The rule is randomized per seed (XOR mask, modulus, labels, and the inputs), so the model
cannot memorize a fixed answer across runs and every seed is a genuinely different
instance — which is what the cost study needs to average over.

Adapted from the "candidate_function" idea on purpose: we never `exec` model-written code
(arbitrary-code-execution risk). Predicting outputs for fixed inputs tests the same skill
safely — the model must generalize one rule that satisfies every query at once.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass

from agentic_loop import Candidate, History, Problem

_LIST_RE = re.compile(r"\[[^\]]*\]")
_LABEL_PAIRS = (("FIZZ", "BUZZ"), ("ALPHA", "BETA"), ("HI", "LO"), ("ON", "OFF"), ("PING", "PONG"))


def _make_rule(seed: int | None) -> tuple[Callable[[int], str], random.Random]:
    """Build a randomized hidden rule: XOR by a mask, then a modulo test picks the label.

    XOR (not addition) means a linear ax+b guess breaks on inputs whose bits overlap the
    mask; the modulo on the *transformed* value hides a second layer. Both parameters and
    the labels are seeded, so each seed is a different rule.
    """
    rng = random.Random(seed)
    mask = rng.randint(16, 63)
    modulus = rng.choice((3, 4, 5))
    label_zero, label_rest = rng.choice(_LABEL_PAIRS)

    def rule(value: int) -> str:
        transformed = value ^ mask
        label = label_zero if transformed % modulus == 0 else label_rest
        return f"{transformed}_{label}"

    return rule, rng


@dataclass(frozen=True)
class ReverseEngineering(Problem):
    """One instance: shown examples, the query inputs, and their gate-only expected outputs."""

    examples: tuple[tuple[int, str], ...]  # (input, output) pairs shown to the model
    queries: tuple[int, ...]  # inputs the model must predict, in order
    expected: tuple[str, ...]  # correct outputs for `queries` — gate-only

    def render_prompt(self, history: History) -> str:
        shown = "\n".join(f"  f({i}) = {o!r}" for i, o in self.examples)
        tries = "\n".join(f"{c} -> {fb}" for c, fb in history) or "(brak)"
        return (
            "Ukryta funkcja f(int) -> str działa wg jednej reguły. Przykłady:\n"
            f"{shown}\n\n"
            f"Podaj wyniki f dla wejść (w tej kolejności): {list(self.queries)}\n"
            f"Poprzednie próby i feedback:\n{tries}\n\n"
            'Zwróć WYŁĄCZNIE listę JSON stringów, np. ["12_AAA", "5_BBB"]. Bez wyjaśnień.'
        )

    def parse_answer(self, answer: str) -> list[Candidate]:
        candidates: list[Candidate] = []
        for match in _LIST_RE.findall(answer):
            try:
                parsed = json.loads(match)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
                candidates.append(parsed)
        return candidates


def verify(problem: ReverseEngineering, candidate: Candidate) -> tuple[bool, str]:
    """The external gate: compare predicted outputs to the hidden rule's outputs."""
    if not (isinstance(candidate, list) and all(isinstance(s, str) for s in candidate)):
        return False, "kandydat musi być listą stringów"
    if len(candidate) != len(problem.queries):
        return False, f"oczekiwano {len(problem.queries)} wyników, jest {len(candidate)}"
    wrong = [
        {"wejście": q, "twój_wynik": got}
        for q, got, exp in zip(problem.queries, candidate, problem.expected, strict=True)
        if got != exp
    ]
    if not wrong:
        return True, "OK"
    return False, f"błędne: {wrong}"


def generate(n: int = 12, seed: int | None = None) -> ReverseEngineering:
    """Random instance: a seeded hidden rule, revealing examples (including f(0), which
    exposes the mask), and deliberately hard queries (negative and large, where a linear
    guess breaks). `n` is unused."""
    rule, rng = _make_rule(seed)

    # Examples: always include 0 (0 ^ mask == mask, the strongest hint), then add inputs
    # until both labels are demonstrated and we have at least six examples.
    pool = list(range(1, 40))
    rng.shuffle(pool)
    example_inputs = [0]
    for value in pool:
        if len(example_inputs) >= 6 and len({rule(i).split("_")[1] for i in example_inputs}) == 2:
            break
        example_inputs.append(value)

    # Queries: a mix of negative and large inputs, disjoint from the examples.
    queries = (*rng.sample(range(-9, 0), 2), *rng.sample(range(64, 2000), 3))

    return ReverseEngineering(
        examples=tuple((i, rule(i)) for i in example_inputs),
        queries=queries,
        expected=tuple(rule(i) for i in queries),
    )


def describe(problem: ReverseEngineering) -> dict[str, object]:
    """Full scale of the puzzle, for the log: the shown examples plus the queries and their
    expected outputs — which the model must infer and never sees. Lets the analyst judge
    how revealing the examples were."""
    return {
        "kind": "reverse_engineering",
        "fully_observable": False,
        "examples": [[i, o] for i, o in problem.examples],
        "queries": list(problem.queries),
        "expected": list(problem.expected),
        "n_examples": len(problem.examples),
        "n_queries": len(problem.queries),
    }
