"""Grid pathfinder: walk a bot from start to goal across a grid with *hidden* walls.

The best shape for an agentic loop: the model physically cannot solve it in one shot
because it has no wall data. It must probe — propose a path, learn where a wall is from
the gate's feedback, and route around it — accumulating the obstacle map in the history.

Coordinates are (x, y) from the top-left, moves are R/L/D/U. The gate simulates the path
and, on the first illegal move, reports exactly which wall was hit.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from agentic_loop import Candidate, History, Problem

_LIST_RE = re.compile(r"\[[^\]]*\]")
_MOVES = {"R": (1, 0), "L": (-1, 0), "D": (0, 1), "U": (0, -1)}
_WALL_DENSITY = 0.30  # fraction of off-path cells turned into walls


@dataclass(frozen=True)
class Maze(Problem):
    """A grid with hidden walls. A guaranteed wall-free path from start to goal exists."""

    width: int
    height: int
    start: tuple[int, int]
    goal: tuple[int, int]
    walls: frozenset[tuple[int, int]]

    def render_prompt(self, history: History) -> str:
        tries = "\n".join(f"{c} -> {fb}" for c, fb in history) or "(brak)"
        return (
            f"Jesteś robotem na siatce {self.width}x{self.height}, współrzędne (x, y) od 0.\n"
            f"Start: {self.start}. Cel: {self.goal}. Ściany są UKRYTE — poznasz je wpadając.\n"
            "Ruchy: R (x+1), L (x-1), D (y+1), U (y-1).\n"
            f"Poprzednie trasy i feedback:\n{tries}\n\n"
            'Podaj trasę jako listę JSON, np. ["R", "R", "D", "D"]. Bez wyjaśnień.'
        )

    def parse_answer(self, answer: str) -> list[Candidate]:
        routes: list[Candidate] = []
        for match in _LIST_RE.findall(answer):
            try:
                parsed = json.loads(match)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list) and all(isinstance(m, str) for m in parsed):
                routes.append(parsed)
        return routes


def verify(problem: Maze, candidate: Candidate) -> tuple[bool, str]:
    """The external gate: simulate the route, reporting the first wall or off-grid step."""
    if not (isinstance(candidate, list) and all(isinstance(m, str) for m in candidate)):
        return False, 'kandydat musi być listą ruchów, np. ["R", "D"]'
    x, y = problem.start
    for i, move in enumerate(candidate, 1):
        if move not in _MOVES:
            return False, f"ruch {i}: nieznany ruch {move!r} (dozwolone: R, L, U, D)"
        dx, dy = _MOVES[move]
        nx, ny = x + dx, y + dy
        if not (0 <= nx < problem.width and 0 <= ny < problem.height):
            return False, f"ruch {i} ('{move}') wyszedł poza siatkę z ({x},{y})"
        if (nx, ny) in problem.walls:
            return False, f"ruch {i} ('{move}') uderzył w ścianę na ({nx},{ny}) — inną drogą"
        x, y = nx, ny
    if (x, y) == problem.goal:
        return True, "OK"
    return False, f"trasa skończyła się na ({x},{y}), cel to {problem.goal}"


def _carve_path(rng: random.Random, size: int) -> list[tuple[int, int]]:
    """A random simple path start->goal via DFS — usually winding (non-monotone), which is
    what lets it outlast the shortcut-blocking below."""
    start, goal = (0, 0), (size - 1, size - 1)
    stack, visited = [start], {start}
    while stack[-1] != goal:
        x, y = stack[-1]
        nbrs = [
            (x + dx, y + dy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= x + dx < size and 0 <= y + dy < size and (x + dx, y + dy) not in visited
        ]
        if nbrs:
            nxt = rng.choice(nbrs)
            visited.add(nxt)
            stack.append(nxt)
        else:
            stack.pop()
    return stack


def _is_monotone(path: list[tuple[int, int]]) -> bool:
    """True if every step goes right or down — a trivial, shortcut-only path."""
    return all(b[0] >= a[0] and b[1] >= a[1] for a, b in zip(path, path[1:], strict=False))


def _monotone_reach(open_cells: set[tuple[int, int]], size: int) -> dict[tuple[int, int], bool]:
    """Map each cell to whether it is reachable from (0,0) using only R/D moves."""
    reach: dict[tuple[int, int], bool] = {}
    for x in range(size):
        for y in range(size):
            if (x, y) not in open_cells:
                reach[(x, y)] = False
            elif (x, y) == (0, 0):
                reach[(x, y)] = True
            else:
                reach[(x, y)] = reach.get((x - 1, y), False) or reach.get((x, y - 1), False)
    return reach


def _a_monotone_path(reach: dict[tuple[int, int], bool], size: int) -> list[tuple[int, int]]:
    """Reconstruct one R/D-only path to the goal (assumes it is reachable)."""
    cells, (x, y) = [], (size - 1, size - 1)
    while (x, y) != (0, 0):
        cells.append((x, y))
        x, y = (x - 1, y) if reach.get((x - 1, y)) else (x, y - 1)
    return cells


def _block_shortcuts(
    open_cells: set[tuple[int, int]], keep: set[tuple[int, int]], size: int
) -> bool:
    """Wall non-`keep` cells until no R/D-only route to the goal survives. Returns False if
    a monotone route would survive only through `keep` (caller recarves)."""
    goal = (size - 1, size - 1)
    while True:
        reach = _monotone_reach(open_cells, size)
        if not reach[goal]:
            return True
        path = _a_monotone_path(reach, size)
        removable = next((c for c in path if c not in keep and c not in {(0, 0), goal}), None)
        if removable is None:
            return False
        open_cells.discard(removable)


def generate(n: int = 12, seed: int | None = None, size: int = 5) -> Maze:
    """Random maze with NO straight (R/D-only) route — the naive guess always fails — yet
    several winding ways through, so it solves in a few iterations. A winding path is
    carved and kept open; walls are scattered, then enough are added to kill every
    monotone shortcut. `n` is unused; `size` sets the grid."""
    rng = random.Random(seed)
    goal = (size - 1, size - 1)
    for _ in range(200):
        path = _carve_path(rng, size)
        if _is_monotone(path):
            continue  # need a winding guaranteed path to outlast shortcut-blocking
        keep = set(path)
        open_cells = {
            (x, y)
            for x in range(size)
            for y in range(size)
            if (x, y) in keep or rng.random() >= _WALL_DENSITY
        }
        if _block_shortcuts(open_cells, keep, size):
            walls = frozenset(
                (x, y) for x in range(size) for y in range(size) if (x, y) not in open_cells
            )
            return Maze(width=size, height=size, start=(0, 0), goal=goal, walls=walls)
    raise RuntimeError("maze generation failed")  # practically unreachable


def describe(problem: Maze) -> dict[str, object]:
    """Full scale of the grid, for the log: the hidden walls the model must map by probing,
    plus dimensions and endpoints. Lets the analyst see the obstacle density per instance."""
    return {
        "kind": "maze",
        "fully_observable": False,
        "width": problem.width,
        "height": problem.height,
        "start": list(problem.start),
        "goal": list(problem.goal),
        "walls": sorted([list(w) for w in problem.walls]),
        "n_walls": len(problem.walls),
        "cells": problem.width * problem.height,
    }
