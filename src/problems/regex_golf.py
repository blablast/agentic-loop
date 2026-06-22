"""Regex golf: write ONE pattern that captures a hidden rule, not a word list.

The prompt shows only a handful of words; the gate checks a large hidden sample drawn from
the same structural rule (a suffix, prefix, infix, or a doubled letter). Two things make
enumeration impossible. First, the model never sees the hidden words, so it cannot list
them. Second — and this is the subtle part — the gate reports only error COUNTS, never the
offending words: revealing them would let the model harvest the hidden set over iterations
and enumerate it (a classic test-set leak). Only mismatches among the already-shown,
public words are named. So the one thing that satisfies a big unseen sample is the rule.

The rule family and its parameters are randomized per seed, so every run is a different
puzzle — which the cost study needs. The hidden train/test split is the same held-out
principle the article argues for, applied to a regex.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass

from agentic_loop import Candidate, History, Problem

_FENCE_RE = re.compile(r"```(?:[a-zA-Z]+\n)?(.*?)```", re.DOTALL)  # lang tag eaten only before a \n
_INLINE_RE = re.compile(r"`([^`\n]+)`")  # `...` inline code
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
_HIDDEN = 24  # words the gate checks per polarity (large -> enumeration is hopeless)
_SHOWN = 4  # words the prompt reveals per polarity

_WordMaker = Callable[[random.Random], str]


@dataclass(frozen=True)
class RegexGolf(Problem):
    """`positive`/`negative` are the full sets the gate checks; `shown_*` are the small
    subset the prompt reveals."""

    positive: tuple[str, ...]
    negative: tuple[str, ...]
    shown_positive: tuple[str, ...]
    shown_negative: tuple[str, ...]

    def render_prompt(self, history: History) -> str:
        tries = "\n".join(f"{c!r} -> {fb}" for c, fb in history) or "(brak)"
        return (
            "Napisz wyrażenie regularne (Python `re`, fullmatch), które uchwyci REGUŁĘ "
            "stojącą za słowami. Jest testowane na DUŻYM ukrytym zbiorze, więc wyliczenie "
            "pokazanych słów nie przejdzie — musisz trafić w regułę.\n"
            f"  - PASUJE do: {list(self.shown_positive)}\n"
            f"  - NIE pasuje do: {list(self.shown_negative)}\n"
            f"Poprzednie próby i feedback:\n{tries}\n\n"
            "Zwróć WYŁĄCZNIE sam wzorzec w bloku ``` ```. Bez wyjaśnień."
        )

    def parse_answer(self, answer: str) -> list[Candidate]:
        seen: list[Candidate] = []  # dedupe: a fenced one-liner also matches the inline rule
        for raw in _FENCE_RE.findall(answer) + _INLINE_RE.findall(answer):
            pattern = raw.strip()
            if pattern and pattern not in seen:
                seen.append(pattern)
        return seen


def verify(problem: RegexGolf, candidate: Candidate) -> tuple[bool, str]:
    """The external gate: compile the pattern and check both full example sets.

    Feedback is COUNTS only (plus mismatches among the shown public words). It never names
    a hidden word — that would leak the held-out set and let the model enumerate it.
    """
    if not isinstance(candidate, str):
        return False, "kandydat musi być wzorcem (string)"
    try:
        pattern = re.compile(candidate)
    except re.error as e:
        return False, f"niepoprawny regex: {e}"
    missed = [w for w in problem.positive if not pattern.fullmatch(w)]  # false negatives
    caught = [w for w in problem.negative if pattern.fullmatch(w)]  # false positives
    if not missed and not caught:
        return True, "OK"
    msg = (
        f"łapie {len(caught)} słów, których nie powinno; pomija {len(missed)} z tych, które powinno"
    )
    shown_missed = [w for w in problem.shown_positive if w in missed]
    shown_caught = [w for w in problem.shown_negative if w in caught]
    notes = []
    if shown_missed:
        notes.append(f"pomijasz pokazane {shown_missed}")
    if shown_caught:
        notes.append(f"łapiesz pokazane {shown_caught}")
    if notes:
        msg += " (" + "; ".join(notes) + ")"
    return False, msg


def _word(rng: random.Random, lo: int = 3, hi: int = 8) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(lo, hi)))


def _word_no_doubles(rng: random.Random, lo: int = 4, hi: int = 8) -> str:
    """A word with no two adjacent equal letters (so it can never match a doubled-letter rule)."""
    n = rng.randint(lo, hi)
    out = [rng.choice(_ALPHABET)]
    while len(out) < n:
        c = rng.choice(_ALPHABET)
        if c != out[-1]:
            out.append(c)
    return "".join(out)


def _build_family(rng: random.Random) -> tuple[str, _WordMaker, _WordMaker]:
    """Pick a structural rule family and randomize its parameter. Returns the canonical
    solution regex plus makers for positive and (near-miss) negative words."""
    kind = rng.choice(("suffix", "prefix", "infix", "double"))
    if kind == "suffix":
        s = rng.choice(("ck", "sh", "ng", "ly", "th", "st", "rd"))

        def suffix_pos(r: random.Random) -> str:
            return _word(r) + s

        def suffix_neg(r: random.Random) -> str:  # random word, or the suffix not at the end
            return r.choice((_word(r), s + _word(r, 1, 4)))

        return f"[a-z]*{s}", suffix_pos, suffix_neg
    if kind == "prefix":
        p = rng.choice(("pre", "un", "re", "sub", "over", "mis"))

        def prefix_pos(r: random.Random) -> str:
            return p + _word(r)

        def prefix_neg(r: random.Random) -> str:
            return r.choice((_word(r), _word(r, 1, 4) + p))

        return f"{p}[a-z]*", prefix_pos, prefix_neg
    if kind == "infix":
        inf = rng.choice(("zz", "qu", "xy", "kk", "wv"))

        def infix_pos(r: random.Random) -> str:
            return _word(r) + inf + _word(r)

        def infix_neg(r: random.Random) -> str:
            return _word(r)

        return f"[a-z]*{inf}[a-z]*", infix_pos, infix_neg

    # "double": a repeated adjacent letter anywhere in the word
    def double_pos(r: random.Random) -> str:
        w = _word(r)
        i = r.randrange(len(w))
        return w[:i] + w[i] + w[i:]  # duplicate one letter -> guaranteed adjacent pair

    def double_neg(r: random.Random) -> str:
        return _word_no_doubles(r)

    return r"[a-z]*([a-z])\1[a-z]*", double_pos, double_neg


def generate(n: int = 12, seed: int | None = None) -> RegexGolf:
    """Random structural puzzle: a seeded rule family, a large hidden test set, and a small
    shown subset. `n` is unused — difficulty is the hidden structure, not instance size."""
    rng = random.Random(seed)
    solution, make_pos, make_neg = _build_family(rng)
    sol = re.compile(solution)

    positive: set[str] = set()
    while len(positive) < _HIDDEN:
        w = make_pos(rng)
        if sol.fullmatch(w):  # always true by construction; a guard against edge cases
            positive.add(w)
    negative: set[str] = set()
    while len(negative) < _HIDDEN:
        w = make_neg(rng)
        if not sol.fullmatch(w) and w not in positive:  # a real near-miss, not an accidental match
            negative.add(w)

    pos_t = tuple(sorted(positive))
    neg_t = tuple(sorted(negative))
    return RegexGolf(
        positive=pos_t,
        negative=neg_t,
        shown_positive=tuple(rng.sample(pos_t, _SHOWN)),
        shown_negative=tuple(rng.sample(neg_t, _SHOWN)),
    )


def describe(problem: RegexGolf) -> dict[str, object]:
    """Full scale of the puzzle, for the log: the large hidden word sets the model never
    sees, next to the small shown subset. Reveals the true scale to the analyst."""
    return {
        "kind": "regex_golf",
        "fully_observable": False,
        "shown_positive": list(problem.shown_positive),
        "shown_negative": list(problem.shown_negative),
        "hidden_positive": list(problem.positive),
        "hidden_negative": list(problem.negative),
        "n_shown": len(problem.shown_positive) + len(problem.shown_negative),
        "n_hidden": len(problem.positive) + len(problem.negative),
    }
