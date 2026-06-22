"""Deterministic demo - runs without an API key.

Run with `uv run python examples/run_demo.py` — no API key needed.
Shows three things:
  1. the loop harness works (brute-force maker),
  2. feedback works (heuristic maker uses the gate's feedback),
  3. why the gate is external (the maker's "confidence" is wrong, the verifier is not).
"""

from agentic_loop import LoopView, loop
from problems.subset_sum import BruteForceMaker, HeuristicMaker, generate, verify

SEED = 7  # fixed seed -> reproducible demo; remove to randomize each run


def header(text):
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def show_attempt(i, candidate, ok, feedback):
    print(f"[iter {i:2}] {candidate}  ->  {feedback}")


def main():
    problem = generate(n=12, seed=SEED)
    print(f"Set S  = {list(problem.numbers)}")
    print(f"Target = {problem.target}   (find a subset of S summing to {problem.target})")

    header("1) Brute-force maker (deterministic proof the loop works)")
    solution, iters = loop(problem, BruteForceMaker(), verify, max_iters=5000)
    print(f"Solution: {solution}  (after {iters} attempts)")
    print(f"Independent check: verify -> {verify(problem, solution)[1]}")

    header("2) Heuristic maker (stands in for an LLM: uses the gate's feedback)")
    solution, iters = loop(
        problem,
        HeuristicMaker(seed=3),
        verify,
        max_iters=40,
        view=LoopView(on_verdict=show_attempt),
    )
    if solution:
        print(f"\nSUCCESS after {iters} iterations: {solution}")
    else:
        print(f"\nNo solution within the limit of {iters}")

    header("3) Why the gate must be EXTERNAL")
    wrong = [problem.numbers[0], problem.numbers[1]]  # almost certainly a wrong sum
    print(f'Maker "claims" with confidence: solution is {wrong} (model_sure = True)')
    ok, feedback = verify(problem, wrong)
    print(f"External gate says: ok = {ok}  ({feedback})")
    print("Takeaway: the maker's self-assessment does not matter - the objective test does.")


if __name__ == "__main__":
    main()
