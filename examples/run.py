"""Run one problem end-to-end against a model (Gemini in the cloud, or a local Ollama
model via --backend ollama). Thinking and the answer stream live, each in its own color;
the verifier's verdict is shown per attempt.

    export GEMINI_API_KEY=...                 # or put it in .env (Gemini backend)
    uv run python examples/run.py --help                    # all options
    uv run python examples/run.py --problem subset_sum -n 24 --thinking high
    uv run python examples/run.py --problem regex_golf --thinking high   # auto-logs to logs/
    uv run python examples/run.py --backend ollama --model qwen3:4b --problem maze

Logging is ON by default: each run writes a JSONL to logs/<problem_params_timestamp>.jsonl
(token usage per attempt + a run summary). `--nolog` disables it; `--log-transcripts` adds
a full prompt+thinking+answer file alongside.

Maker = a generic model (Gemini or Ollama). Gate = the chosen problem's `verify`. No MCP —
the verifier is not a tool the model calls, just an external test in the loop's code.
"""

import argparse
import os
import random
import sys
import uuid
from datetime import datetime

try:  # load GEMINI_API_KEY from a local .env if present
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional — env vars still work without it
    pass

from console import Console, StreamView  # presentation layer (sibling module)
from fanout import tee_hooks, tee_view  # compose console + logger into one seam
from pricing import PRICES, PRICES_AS_OF  # price table injected into the logger (sibling)
from session_log import RunConfig, SessionLogger  # JSONL token/usage logger (sibling)

from agentic_loop import (  # the display-agnostic core
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    GeminiMaker,
    LoopView,
    Maker,
    OllamaMaker,
    StreamHooks,
    loop,
)
from problems import REGISTRY, get  # the problem registry (pick one by name)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--problem",
        default="subset_sum",
        choices=sorted(REGISTRY),
        help="which problem to solve (default subset_sum)",
    )
    p.add_argument("-n", type=int, default=12, help="instance size; bigger is harder (default 12)")
    p.add_argument(
        "--backend",
        default="gemini",
        choices=["gemini", "ollama"],
        help="generation backend (default gemini)",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"model id (default {DEFAULT_MODEL}; for ollama e.g. gemma4:31b)",
    )
    p.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama server URL (default {DEFAULT_OLLAMA_HOST})",
    )
    p.add_argument(
        "--thinking",
        default="high",
        choices=["minimal", "low", "medium", "high"],
        help="Gemini thinking level; for ollama only 'minimal' (off) vs other (on) matters",
    )
    p.add_argument("--max-iters", type=int, default=12, help="hard stop on attempts (default 12)")
    p.add_argument("--seed", type=int, default=None, help="task seed; omit for a random task")
    p.add_argument("--nolog", action="store_true", help="disable logging (on by default)")
    p.add_argument(
        "--log-dir",
        metavar="DIR",
        default="logs",
        help="directory for auto-named JSONL logs (default logs/)",
    )
    p.add_argument(
        "--log-transcripts",
        action="store_true",
        help="also write full prompt+thinking+answer to a sibling .transcripts file",
    )
    return p.parse_args()


def _log_filename(args: argparse.Namespace, stamp: str) -> str:
    """Auto-name a log from the problem, the call's parameters, and the launch timestamp."""
    seed = args.seed  # always concrete (resolved in main), so the log name is reproducible
    model = args.model.replace("/", "-")
    return (
        f"{args.problem}_{model}_{args.thinking}"
        f"_n{args.n}_seed{seed}_it{args.max_iters}_{stamp}.jsonl"
    )


def build_logger(args: argparse.Namespace, description: dict[str, object]) -> SessionLogger:
    from importlib.metadata import PackageNotFoundError, version

    try:
        sdk_version = version("google-genai")
    except PackageNotFoundError:
        sdk_version = "unknown"
    config = RunConfig(
        problem=args.problem,
        model=args.model,
        thinking_level=args.thinking,
        seed=args.seed,
        max_iters=args.max_iters,
        sdk_version=sdk_version,
    )
    os.makedirs(args.log_dir, exist_ok=True)
    path = os.path.join(args.log_dir, _log_filename(args, datetime.now().strftime("%Y%m%d-%H%M%S")))
    transcript_path = None
    if args.log_transcripts:
        base, ext = os.path.splitext(path)
        transcript_path = f"{base}.transcripts{ext}"
    return SessionLogger(
        path,
        uuid.uuid4().hex[:12],
        config,
        description=description,
        transcript_path=transcript_path,
        prices=PRICES,
        prices_as_of=PRICES_AS_OF,
    )


def require_api_key():
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit(
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY) — e.g. `cp .env.example .env` and fill it in."
        )


def main():
    args = parse_args()
    if args.seed is None:
        args.seed = random.randrange(2**32)  # resolve now so the run is reproducible from its log
    if args.backend == "gemini":
        require_api_key()

    console = Console()
    view = StreamView(console)

    spec = get(args.problem)
    problem = spec.generate(args.n, args.seed)
    console.line(
        f"Problem = {spec.name}   backend = {args.backend}   "
        f"model = {args.model}   thinking = {args.thinking}",
        "bold",
    )  # the instance itself appears in the streamed prompt below

    def on_attempt(i):
        console.line(f"\niteration {i}", "bold", "yellow")

    def on_verdict(i, candidate, ok, feedback):
        view.end()  # close the streamed answer section
        console.line(f"verdict: {candidate} -> {feedback}", "green" if ok else "red")

    console_hooks = StreamHooks(
        on_prompt=lambda text: view.block("prompt", text), on_token=view.token
    )
    console_view = LoopView(on_attempt=on_attempt, on_verdict=on_verdict)

    # console and logger are independent observers of the same events — tee into one seam.
    # Logging is on by default (auto-named file); --nolog turns it off.
    logger = None if args.nolog else build_logger(args, spec.describe(problem))
    if logger:
        console.line(f"Log = {logger.path}", "dim")
    hooks = tee_hooks(console_hooks, logger.as_hooks()) if logger else console_hooks
    loop_view = tee_view(console_view, logger.as_view()) if logger else console_view

    catchable: tuple[type[BaseException], ...] = (OSError,)  # Ollama URLError is an OSError
    maker: Maker
    if args.backend == "ollama":
        maker = OllamaMaker(
            model=args.model,
            host=args.ollama_host,
            think=(args.thinking != "minimal"),  # only on/off maps to Ollama
            hooks=hooks,
        )
    else:
        from google.genai.errors import APIError  # local import: --help works without a call

        catchable = (APIError, OSError)
        maker = GeminiMaker(model=args.model, thinking_level=args.thinking, hooks=hooks)

    try:
        solution, iters = loop(
            problem, maker, spec.verify, max_iters=args.max_iters, view=loop_view
        )
    except catchable as e:
        message = getattr(e, "message", None) or str(e)
        if args.backend == "gemini" and "model" in message.lower():
            message += "\nTip: model ids are lowercase, e.g. `gemini-3.5-flash` — check --model."
        if args.backend == "ollama":
            message += f"\nTip: is Ollama reachable at {args.ollama_host}? Check --model."
        sys.exit(f"\n{args.backend} backend error: {message}")
    finally:
        if logger:
            logger.close()

    if solution:
        feedback = spec.verify(problem, solution)[1]
        console.line(
            f"\nSUCCESS after {iters} iterations: {solution}  ({feedback})",
            "bold",
            "green",
        )
    else:
        console.line(
            f"\nNo solution in {iters} iterations — the gate let no wrong answer through.",
            "bold",
            "red",
        )


if __name__ == "__main__":
    main()
