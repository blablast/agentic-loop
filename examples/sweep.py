"""Benchmark sweep: run the problems across a grid of (backend, model, thinking) x seeds,
one logged run per cell, so analyze_logs.py can compute $/success.

Each cell is one invocation of run.py (its sibling), which writes its own JSONL log
— no maker/logging logic is duplicated here, we just drive the existing single-run entry
point across the grid. Crucially, every cell uses the SAME seed list, so models face
identical instances (a paired comparison). That is the whole point.

Fill CELLS below with the models you have (e.g. after `ollama list`); seeds default to 0-9.
Then:  uv run python examples/sweep.py          (add --dry-run to preview the plan first)
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from run import safe_filename  # same sanitizer the runner uses, so resume matches real names

from problems import REGISTRY


@dataclass(frozen=True)
class Cell:
    """One model to sweep, and which thinking levels to try for it."""

    backend: str  # "gemini" | "ollama"
    model: str
    thinkings: tuple[str, ...]  # e.g. ("minimal", "high"); for ollama only on/off matters


# --- THE GRID --------------------------------------------------------------------------
# Fill this with your models. One Cell per model. Examples (uncomment / edit):
CELLS: list[Cell] = [
    # gemma4 family — the controlled size ladder with thinking off vs on:
    Cell("ollama", "gemma4:latest", ("minimal", "high")),  # 8B
    Cell("ollama", "gemma4:26b", ("minimal", "high")),
    Cell("ollama", "gemma4:31b", ("minimal", "high")),
    # tiny floor (where the loop and best-of-N earn their keep):
    Cell("ollama", "qwen3:4b", ("minimal", "high")),
    # fast small model for the latency axis (LFM, thinking-capable):
    Cell("ollama", "lfm2.5:latest", ("minimal", "high")),
    # reasoning specialists (R1 always reasons -> on only):
    Cell("ollama", "deepseek-r1:14b", ("high",)),
    Cell("ollama", "deepseek-r1:32b", ("high",)),  # still pulling — will be ready
    # cross-family contrast:
    Cell("ollama", "qwen3:14b", ("minimal", "high")),
    Cell("ollama", "gpt-oss:latest", ("minimal", "high")),
    #
    # NOTE: our OllamaMaker maps thinking to on/off (think=bool), so for ollama use only
    # "minimal" (off) vs one "on" value — "low"/"medium"/"high" all collapse to think=True.
    #
    # Optional extra reasoning/large points (uncomment to add):
    # Cell("ollama", "qwen3.6:27b", ("minimal", "high")),
    # Cell("ollama", "qwen3.6:latest", ("minimal", "high")),  # 36B
    #
    # Gemini (cloud, paid) — uncomment for the real $/success numbers (levels DO differ):
    # Cell("gemini", "gemini-3.5-flash", ("minimal", "low", "high")),
    # Cell("gemini", "gemini-3.1-flash-lite", ("minimal", "high")),
]

DEFAULT_SEEDS = "0-9"
RUNNER = Path(__file__).with_name("run.py")


def parse_seeds(spec: str) -> list[int]:
    """Parse '0-9' or '0,1,2' or '0-4,7' into a list of seeds."""
    seeds: list[int] = []
    for part in (p.strip() for p in spec.split(",")):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def already_logged(
    log_dir: str, problem: str, model: str, thinking: str, n: int, max_iters: int, seed: int
) -> bool:
    """True if a COMPLETED run (a log carrying a `summary` record) already exists for this
    cell. Lets an interrupted sweep resume: re-running the same command skips done cells.
    The cell's parameters (incl. n and max_iters) are in the filename, so changing them
    forces a fresh run; for changed gate semantics (e.g. tolerance), use a fresh --log-dir.
    """
    prefix = f"{problem}_{safe_filename(model)}_{thinking}_n{n}_seed{seed}_it{max_iters}_"
    for path in glob.glob(os.path.join(log_dir, prefix + "*.jsonl")):
        if path.endswith(".transcripts.jsonl"):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                if any('"record_type": "summary"' in line for line in handle):
                    return True
        except OSError:
            continue
    return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default=DEFAULT_SEEDS, help=f"seeds (default {DEFAULT_SEEDS})")
    p.add_argument(
        "--problems",
        default=",".join(sorted(REGISTRY)),
        help="comma-separated problems (default: all)",
    )
    p.add_argument("-n", type=int, default=22, help="instance size (subset_sum; default 22)")
    p.add_argument("--max-iters", type=int, default=12, help="hard stop per run (default 12)")
    p.add_argument("--ollama-host", default=None, help="Ollama server URL (for ollama cells)")
    p.add_argument("--log-dir", default="logs", help="where runs write their JSONL (default logs/)")
    p.add_argument("--log-transcripts", action="store_true", help="also save full transcripts")
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="re-run every cell (default: skip cells already logged in --log-dir)",
    )
    p.add_argument("--dry-run", action="store_true", help="print the planned runs, do not execute")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not CELLS:
        sys.exit("CELLS is empty — add your (backend, model, thinkings) at the top of sweep.py.")

    seeds = parse_seeds(args.seeds)
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    unknown = [p for p in problems if p not in REGISTRY]
    if unknown:
        sys.exit(f"unknown problem(s): {', '.join(unknown)}; known: {', '.join(sorted(REGISTRY))}")

    plan = [
        (cell, thinking, problem, seed)
        for cell in CELLS
        for thinking in cell.thinkings
        for problem in problems
        for seed in seeds
    ]
    print(
        f"Planned {len(plan)} runs: {len(CELLS)} models x thinkings "
        f"x {len(problems)} problems x {len(seeds)} seeds -> {args.log_dir}/"
    )

    ran = skipped = failures = 0
    for i, (cell, thinking, problem, seed) in enumerate(plan, 1):
        label = (
            f"[{i}/{len(plan)}] {cell.backend}:{cell.model} think={thinking} {problem} seed={seed}"
        )
        if (
            not args.no_resume
            and not args.dry_run
            and already_logged(
                args.log_dir, problem, cell.model, thinking, args.n, args.max_iters, seed
            )
        ):
            skipped += 1
            print(f"{label}  (skip: already logged)", flush=True)
            continue
        cmd = [
            sys.executable,
            str(RUNNER),
            "--backend",
            cell.backend,
            "--model",
            cell.model,
            "--thinking",
            thinking,
            "--problem",
            problem,
            "-n",
            str(args.n),
            "--max-iters",
            str(args.max_iters),
            "--seed",
            str(seed),
            "--log-dir",
            args.log_dir,
        ]
        if args.log_transcripts:
            cmd.append("--log-transcripts")
        if cell.backend == "ollama" and args.ollama_host:
            cmd += ["--ollama-host", args.ollama_host]

        if args.dry_run:
            print(label)
            continue
        print(label, flush=True)
        ran += 1
        if subprocess.run(cmd).returncode != 0:  # the runner streams output + writes its log
            failures += 1
            print("  ! run failed (see error above)", flush=True)

    if not args.dry_run:
        print(
            f"\nDone. ran {ran} ({ran - failures} ok, {failures} failed), "
            f"skipped {skipped} already-logged, of {len(plan)} planned."
        )
        print(f"Analyze:  uv run python examples/analyze_logs.py {args.log_dir}")


if __name__ == "__main__":
    main()
