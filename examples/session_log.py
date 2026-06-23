"""Session logger — a display-agnostic observer that records per-attempt token usage and
a run summary to JSONL, for offline cost analysis ($/success across model×thinking).

It lives in examples/ (the observation layer) and wires through the same injection seams
as the console — `LoopView` (on_attempt/on_verdict) and `StreamHooks` (on_prompt/on_token/
on_usage). The core never knows it exists.

Hard rules baked in:
  - log RAW token counts; a derived `cost_usd` is added to the summary only when a price
    table is injected, alongside the prices + as-of date that produced it — so the cost
    stays reproducible and auditable even though prices change,
  - keep the light metrics JSONL separate from full transcripts (thinking text is huge);
    transcripts are opt-in via a second file,
  - stamp every record with `schema_version`; put the full run config in the summary,
    so a run is reproducible from the log alone.

Correlation: one maker call == one iteration. `on_attempt(i)` opens an iteration buffer,
`on_usage` fills its token counts, `on_verdict` appends each candidate's verdict (a batch
yields several). A record is flushed at the next `on_attempt` and at `close()`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import IO, Any

from agentic_loop import LoopView, StreamHooks, Usage

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to reproduce a run from its log alone."""

    problem: str
    model: str
    thinking_level: str
    seed: int | None
    max_iters: int
    sdk_version: str


class SessionLogger:
    """Writes one `attempt` record per iteration and a final `summary` record. Open in
    append mode and flush after every write, so a crash mid-run still leaves valid JSONL.
    """

    def __init__(
        self,
        path: str,
        run_id: str,
        config: RunConfig,
        *,
        description: dict[str, Any] | None = None,
        transcript_path: str | None = None,
        prices: dict[str, tuple[float, float]] | None = None,
        prices_as_of: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path
        self._config = config
        self._run_id = run_id
        self._prices = prices  # model -> (input $/1M, output $/1M); injected, never imported
        self._prices_as_of = prices_as_of
        self._clock = clock
        self._metrics: IO[str] = open(path, "a", encoding="utf-8")  # noqa: SIM115
        self._transcripts: IO[str] | None = (
            open(transcript_path, "a", encoding="utf-8") if transcript_path else None  # noqa: SIM115
        )
        if description is not None:
            # The instance's full scale (incl. parts hidden from the model). Written first,
            # so a crash still leaves the analyst the problem's size next to the costs.
            self._write(
                self._metrics,
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "problem",
                    "run_id": run_id,
                    "problem": config.problem,
                    "seed": config.seed,
                    "description": description,
                },
            )
        self._iter: int | None = None
        self._candidates: list[dict[str, Any]] = []
        self._usage: Usage | None = None
        self._prompt: str | None = None
        self._stream: list[tuple[str, str]] = []
        self._totals: dict[str, float] = {
            "prompt": 0,
            "candidate": 0,
            "thoughts": 0,
            "total": 0,
            "latency_ms": 0.0,
        }
        self._solved = False
        self._iters_run = 0

    # --- wiring into the two injection seams ---------------------------------

    def as_view(self) -> LoopView:
        return LoopView(on_attempt=self.on_attempt, on_verdict=self.on_verdict)

    def as_hooks(self) -> StreamHooks:
        return StreamHooks(on_prompt=self.on_prompt, on_token=self.on_token, on_usage=self.on_usage)

    # --- callbacks -----------------------------------------------------------

    def on_attempt(self, i: int) -> None:
        self._flush()  # the previous iteration is complete
        self._iter = i
        self._iters_run = i
        self._candidates = []
        self._usage = None
        self._prompt = None
        self._stream = []

    def on_prompt(self, text: str) -> None:
        self._prompt = text

    def on_token(self, kind: str, text: str) -> None:
        if self._transcripts is not None:
            self._stream.append((kind, text))

    def on_usage(self, usage: Usage) -> None:
        self._usage = usage

    def on_verdict(self, i: int, candidate: Any, ok: bool, feedback: str) -> None:
        self._candidates.append({"candidate": candidate, "ok": ok, "feedback": feedback})
        if ok:
            self._solved = True

    # --- writing -------------------------------------------------------------

    def _flush(self) -> None:
        if self._iter is None:
            return
        usage = self._usage
        self._write(
            self._metrics,
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "attempt",
                "run_id": self._run_id,
                "ts": self._clock(),
                "problem": self._config.problem,
                "model": self._config.model,
                "thinking_level": self._config.thinking_level,
                "seed": self._config.seed,
                "iter": self._iter,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "candidate_tokens": usage.candidate_tokens if usage else 0,
                "thoughts_tokens": usage.thoughts_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "latency_ms": round(usage.latency_ms, 1) if usage else 0,
                "candidates": self._candidates,
                "solved_here": any(c["ok"] for c in self._candidates),
            },
        )
        if usage:
            self._totals["prompt"] += usage.prompt_tokens
            self._totals["candidate"] += usage.candidate_tokens
            self._totals["thoughts"] += usage.thoughts_tokens
            self._totals["total"] += usage.total_tokens
            self._totals["latency_ms"] += usage.latency_ms
        if self._transcripts is not None:
            self._write(
                self._transcripts,
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "transcript",
                    "run_id": self._run_id,
                    "iter": self._iter,
                    "prompt": self._prompt,
                    "thinking": "".join(t for k, t in self._stream if k == "thinking"),
                    "answer": "".join(t for k, t in self._stream if k == "answer"),
                },
            )
        self._iter = None

    def close(self) -> None:
        self._flush()
        cost_usd, cost_prices = self._cost()
        self._write(
            self._metrics,
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "summary",
                "run_id": self._run_id,
                "problem": self._config.problem,
                "model": self._config.model,
                "thinking_level": self._config.thinking_level,
                "seed": self._config.seed,
                "max_iters": self._config.max_iters,
                "solved": self._solved,
                "iters": self._iters_run,
                "prompt_tokens_total": self._totals["prompt"],
                "candidate_tokens_total": self._totals["candidate"],
                "thoughts_tokens_total": self._totals["thoughts"],
                "total_tokens_total": self._totals["total"],
                "latency_ms_total": round(self._totals["latency_ms"], 1),  # warm gen, load-excl.
                "cost_usd": cost_usd,  # derived; null if the model's price is unknown
                "cost_prices": cost_prices,  # the price snapshot that produced cost_usd
                "sdk_version": self._config.sdk_version,
                "config": asdict(self._config),
            },
        )
        self._metrics.close()
        if self._transcripts is not None:
            self._transcripts.close()

    def _cost(self) -> tuple[float | None, dict[str, Any] | None]:
        """Run cost in USD, plus the price snapshot used. None when no price table is
        injected or the model is unknown — the cost stays auditable and reproducible
        (thinking tokens bill at the output rate)."""
        model = self._config.model
        if not self._prices or model not in self._prices:
            return None, None
        input_price, output_price = self._prices[model]
        output_tokens = self._totals["candidate"] + self._totals["thoughts"]
        usd = (self._totals["prompt"] * input_price + output_tokens * output_price) / 1_000_000
        snapshot = {
            "as_of": self._prices_as_of,
            "input_per_1m": input_price,
            "output_per_1m": output_price,
        }
        return round(usd, 6), snapshot

    def __enter__(self) -> SessionLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @staticmethod
    def _write(stream: IO[str], record: dict[str, Any]) -> None:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        stream.flush()
