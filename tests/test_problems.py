"""Tests for every concrete problem.

Two layers:
  - a shared CONTRACT parametrized over the whole REGISTRY, so the invariants every
    problem must hold (reproducible generation, a deterministic gate, a JSON-safe
    description, an accepted solution) are checked uniformly and a new problem inherits
    them for free;
  - per-problem SPECIFICS, including the randomized generators (which fixed instances
    never exercise) and each gate's concrete feedback.

This file folds in the former test_subset_sum.py. test_loop.py still covers the generic
core; pricing/session-log tests live with the observation layer.
"""

from __future__ import annotations

import json
import random
import re
from collections import deque
from itertools import combinations
from typing import Any

import pytest

from problems import (
    REGISTRY,
    maze,
    optimization,
    regex_golf,
    reverse_engineering,
    subset_sum,
)
from problems.maze import Maze
from problems.optimization import Optimization
from problems.regex_golf import RegexGolf
from problems.subset_sum import SubsetSum

NAMES = sorted(REGISTRY)


# --- helpers: a known-good solution per problem (also used for determinism) ---------


def _maze_path(m: Maze) -> list[str]:
    """BFS the open cells and return a move list from start to goal."""
    prev: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {m.start: None}
    queue = deque([m.start])
    while queue:
        cur = queue.popleft()
        if cur == m.goal:
            break
        x, y = cur
        for dx, dy, move in ((1, 0, "R"), (-1, 0, "L"), (0, 1, "D"), (0, -1, "U")):
            nxt = (x + dx, y + dy)
            inside = 0 <= nxt[0] < m.width and 0 <= nxt[1] < m.height
            if inside and nxt not in m.walls and nxt not in prev:
                prev[nxt] = (cur, move)
                queue.append(nxt)
    moves: list[str] = []
    cur = m.goal
    while prev[cur] is not None:
        parent, move = prev[cur]  # type: ignore[misc]
        moves.append(move)
        cur = parent
    moves.reverse()
    return moves


def _solve(name: str, seed: int) -> tuple[Any, Any]:
    """Return (instance, a solution the gate accepts) for the given problem and seed."""
    p = REGISTRY[name].generate(12, seed)
    if name == "subset_sum":
        sol: Any = next(
            list(c)
            for r in range(1, len(p.numbers) + 1)
            for c in combinations(p.numbers, r)
            if sum(c) == p.target
        )
    elif name == "optimization":
        sol = [p.target_x, p.target_y]
    elif name == "regex_golf":
        sol = regex_golf._build_family(random.Random(seed))[0]  # canonical rule for this seed
    elif name == "reverse_engineering":
        sol = list(p.expected)
    elif name == "maze":
        sol = _maze_path(p)
    else:  # pragma: no cover - guards against an unregistered problem
        raise AssertionError(f"no solver for {name}")
    return p, sol


# --- shared contract: every registered problem must satisfy these -------------------


@pytest.mark.parametrize("name", NAMES)
def test_generate_is_reproducible(name: str) -> None:
    assert REGISTRY[name].generate(12, 7) == REGISTRY[name].generate(12, 7)


@pytest.mark.parametrize("name", NAMES)
def test_generate_varies_across_seeds(name: str) -> None:
    instances = {REGISTRY[name].generate(12, s) for s in range(8)}
    assert len(instances) > 1  # randomized per seed, not a fixed instance


@pytest.mark.parametrize("name", NAMES)
def test_gate_accepts_a_solution(name: str) -> None:
    problem, solution = _solve(name, 1)
    ok, _ = REGISTRY[name].verify(problem, solution)
    assert ok


@pytest.mark.parametrize("name", NAMES)
def test_verify_is_deterministic(name: str) -> None:
    problem, solution = _solve(name, 1)
    verify = REGISTRY[name].verify
    assert verify(problem, solution) == verify(problem, solution)


@pytest.mark.parametrize("name", NAMES)
def test_parse_answer_always_returns_a_list(name: str) -> None:
    problem = REGISTRY[name].generate(12, 0)
    assert isinstance(problem.parse_answer("garbage with no answer"), list)


@pytest.mark.parametrize("name", NAMES)
def test_describe_is_json_safe_and_tagged(name: str) -> None:
    description = REGISTRY[name].describe(REGISTRY[name].generate(12, 0))
    assert isinstance(description, dict) and description["kind"] == name
    json.dumps(description)  # must be serializable for the log


# --- subset_sum (folded in from the former test_subset_sum.py) ----------------------


def _ss(numbers: list[int], target: int) -> SubsetSum:
    return SubsetSum(tuple(numbers), target)


def test_subset_correct() -> None:
    assert subset_sum.verify(_ss([2, 3, 5, 8], 10), [2, 8])[0]


def test_subset_wrong_sum() -> None:
    ok, feedback = subset_sum.verify(_ss([2, 3, 5, 8], 10), [2, 3])
    assert not ok and "różnica" in feedback


def test_subset_element_not_in_set() -> None:
    ok, feedback = subset_sum.verify(_ss([2, 3, 5], 8), [3, 5, 99])
    assert not ok and "spoza" in feedback


def test_subset_repeated_element() -> None:
    ok, feedback = subset_sum.verify(_ss([3, 5, 8], 16), [8, 8])  # 8+8 hits the sum, but reuses 8
    assert not ok and "powtórzony" in feedback


def test_subset_wrong_type() -> None:
    assert not subset_sum.verify(_ss([1, 2], 3), "1,2")[0]


# --- optimization -------------------------------------------------------------------

_OPT = Optimization(target_x=3.5, target_y=-2.1, a=1.0, b=2.0)


def test_optimization_at_optimum_passes() -> None:
    assert optimization.verify(_OPT, [3.5, -2.1])[0]


def test_optimization_far_point_fails_with_loss() -> None:
    ok, feedback = optimization.verify(_OPT, [0.0, 0.0])
    assert not ok and "loss" in feedback


def test_optimization_rejects_non_point() -> None:
    ok, feedback = optimization.verify(_OPT, [1.0, 2.0, 3.0])
    assert not ok and "punktem" in feedback


def test_optimization_generate_is_anisotropic() -> None:
    # a != b is what forces off-axis probing (kills the "assume a symmetric bowl" shortcut)
    for seed in range(8):
        p = optimization.generate(12, seed)
        assert p.a != p.b


def test_optimization_generate_center_is_not_round() -> None:
    # non-round optima (e.g. -1.76) stop the model from guessing whole/half-integer centers
    centers = [optimization.generate(12, s).target_x for s in range(8)]
    assert any((round(x * 100) % 50) != 0 for x in centers)  # at least one off the .0/.5 grid


# --- regex golf (hidden structural rule, counts-only feedback) ----------------------

# rule = "ends in ck"; only a subset of words is shown to the model
_REGEX = RegexGolf(
    positive=("back", "duck", "rock", "sick"),
    negative=("bad", "cat", "ckaa", "moon"),
    shown_positive=("back", "duck"),
    shown_negative=("bad", "cat"),
)


def test_regex_rule_pattern_passes_full_set() -> None:
    assert regex_golf.verify(_REGEX, r"[a-z]*ck")[0]


def test_regex_feedback_is_counts_only_no_hidden_leak() -> None:
    ok, feedback = regex_golf.verify(_REGEX, r"^(back|duck)$")  # misses hidden rock/sick
    assert not ok
    assert "rock" not in feedback and "sick" not in feedback  # hidden words never named
    assert "pomija" in feedback  # feedback is counts, not the offending words


def test_regex_prompt_shows_only_subset() -> None:
    prompt = _REGEX.render_prompt([])
    assert "back" in prompt and "rock" not in prompt


def test_regex_invalid_pattern_is_caught() -> None:
    ok, feedback = regex_golf.verify(_REGEX, "(")
    assert not ok and "niepoprawny" in feedback


def test_regex_parse_oneline_fence_keeps_prefix() -> None:
    assert _REGEX.parse_answer("```ag(ent|enda|ile)```") == ["ag(ent|enda|ile)"]


def test_regex_generated_canonical_solves_and_shown_is_subset() -> None:
    for seed in range(8):
        p = regex_golf.generate(12, seed)
        assert set(p.shown_positive) <= set(p.positive)
        assert set(p.shown_negative) <= set(p.negative)
        solution, _, _ = regex_golf._build_family(random.Random(seed))  # canonical for this seed
        assert regex_golf.verify(p, solution) == (True, "OK")


def test_regex_generated_feedback_never_leaks_hidden_words() -> None:
    for seed in range(8):
        p = regex_golf.generate(12, seed)
        hidden = (set(p.positive) | set(p.negative)) - set(p.shown_positive) - set(p.shown_negative)
        enum = "(?:" + "|".join(re.escape(w) for w in p.shown_positive) + ")"
        for pattern in ("[a-z]*", "x", enum):
            _, feedback = regex_golf.verify(p, pattern)
            assert not [w for w in hidden if w in feedback]


def test_regex_families_vary_across_seeds() -> None:
    families = {regex_golf._build_family(random.Random(s))[0] for s in range(12)}
    assert len(families) > 1


# --- reverse engineering (XOR + modulo, hidden expected) ----------------------------

_RE = reverse_engineering.generate(seed=0)  # fixed seed -> reproducible test


def test_reverse_correct_outputs_pass() -> None:
    assert reverse_engineering.verify(_RE, list(_RE.expected))[0]


def test_reverse_wrong_outputs_do_not_leak_expected() -> None:
    ok, feedback = reverse_engineering.verify(_RE, ["0_NONE"] * len(_RE.queries))
    assert not ok and "błędne" in feedback
    for expected in _RE.expected:
        assert expected not in feedback  # the gate reveals wrong inputs, never the answer


def test_reverse_wrong_length_fails() -> None:
    ok, feedback = reverse_engineering.verify(_RE, ["1_X"])
    assert not ok and "oczekiwano" in feedback


def test_reverse_linear_guess_fails() -> None:
    # treating the rule as addition (q + mask) instead of XOR must be wrong somewhere
    guess = [f"{q}_X" for q in _RE.queries]
    assert reverse_engineering.verify(_RE, guess)[0] is False


# --- maze (hidden walls) ------------------------------------------------------------

_MAZE = Maze(width=3, height=3, start=(0, 0), goal=(2, 2), walls=frozenset({(1, 0)}))


def test_maze_route_to_goal_passes() -> None:
    assert maze.verify(_MAZE, ["D", "D", "R", "R"])[0]  # avoids the wall at (1,0)


def test_maze_hitting_wall_reports_position() -> None:
    ok, feedback = maze.verify(_MAZE, ["R"])  # (0,0) -> (1,0) is a wall
    assert not ok and "(1,0)" in feedback and "ścian" in feedback


def test_maze_off_grid_fails() -> None:
    ok, feedback = maze.verify(_MAZE, ["U"])  # (0,0) -> (0,-1) leaves the grid
    assert not ok and "poza siatkę" in feedback


def test_maze_unknown_move_fails() -> None:
    ok, feedback = maze.verify(_MAZE, ["X"])
    assert not ok and "nieznany" in feedback


def test_maze_short_route_misses_goal() -> None:
    ok, feedback = maze.verify(_MAZE, ["D"])  # ends at (0,1), not the goal
    assert not ok and "cel" in feedback


def test_maze_has_no_straight_route() -> None:
    # the whole point: a naive R/D-only path must always fail, forcing iteration
    for seed in range(20):
        m = maze.generate(seed=seed)
        straight = ["R"] * (m.width - 1) + ["D"] * (m.height - 1)
        assert not maze.verify(m, straight)[0]


def test_maze_generate_is_always_solvable() -> None:
    for seed in range(20):
        m = maze.generate(seed=seed)
        assert maze.verify(m, _maze_path(m))[0]
