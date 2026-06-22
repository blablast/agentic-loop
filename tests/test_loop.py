"""Tests for the generic harness — independent of any concrete problem.

`test_loop_is_problem_agnostic` is the architectural guarantee: the loop drives a
made-up `Problem` with a made-up maker and verifier, proving it knows nothing about
subset-sum.
"""

from agentic_loop import Problem, loop
from problems.subset_sum import BruteForceMaker, generate, verify


def test_loop_finds_solution_with_brute_force():
    problem = generate(n=8, seed=2)
    solution, _ = loop(problem, BruteForceMaker(), verify, max_iters=5000)
    assert solution is not None
    assert verify(problem, solution)[0]


def test_loop_gives_up_at_max_iters():
    problem = generate(n=12, seed=3)
    solution, iters = loop(problem, lambda p, h: [], verify, max_iters=3)
    assert solution is None and iters == 3


def test_loop_is_problem_agnostic():
    class Fake(Problem):
        def render_prompt(self, history):
            return ""

        def parse_answer(self, answer):
            return []

    def maker(problem, history):
        return [[42]]  # a batch with one candidate

    def gate(problem, candidate):
        return candidate == [42], "fb"

    solution, iters = loop(Fake(), maker, gate, max_iters=2)
    assert solution == [42] and iters == 1
