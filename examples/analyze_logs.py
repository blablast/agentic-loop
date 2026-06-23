"""Analyze sweep logs: group runs by (problem, model, thinking) and report success rate,
median iterations, median tokens (total + thinking), and $/success.

Reads the light metrics JSONL written by the runs (skips the *.transcripts.jsonl). Uses
only the per-run `summary` records.

`$/success` = total USD spent (including failed runs) / number of successes — the real
"cost to get one solution", which is the comparison that matters. Cost is null for local
or unknown-price models (Ollama), shown as '-'; there you compare tokens and success rate.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict
from typing import Any

_COLUMNS = [
    ("problem", "problem", 20),
    ("model", "model", 24),
    ("thinking", "think", 8),
    ("n", "n", 4),
    ("success_rate", "succ%", 6),
    ("med_iters", "med_it", 7),
    ("med_total_tokens", "med_tok", 9),
    ("med_thinking_tokens", "med_think", 10),
    ("med_secs", "med_s", 7),
    ("cost_per_success_usd", "$/succ", 10),
]


def load_summaries(log_dir: str) -> list[dict[str, Any]]:
    """Collect every `summary` record from the metrics JSONL files in `log_dir`."""
    summaries: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.jsonl"))):
        if path.endswith(".transcripts.jsonl"):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("record_type") == "summary":
                    summaries.append(record)
    return summaries


def aggregate(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by (problem, model, thinking_level) and compute the comparison metrics."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for s in summaries:
        groups[(s["problem"], s["model"], s["thinking_level"])].append(s)

    rows: list[dict[str, Any]] = []
    for (problem, model, thinking), runs in sorted(groups.items()):
        n = len(runs)
        n_ok = sum(1 for r in runs if r.get("solved"))
        costs = [r["cost_usd"] for r in runs if r.get("cost_usd") is not None]
        total_cost = sum(costs) if costs else None
        rows.append(
            {
                "problem": problem,
                "model": model,
                "thinking": thinking,
                "n": n,
                "success_rate": n_ok / n,
                "med_iters": statistics.median(r["iters"] for r in runs),
                "med_total_tokens": statistics.median(r["total_tokens_total"] for r in runs),
                "med_thinking_tokens": statistics.median(
                    r.get("thoughts_tokens_total", 0) for r in runs
                ),
                "med_secs": statistics.median(r.get("latency_ms_total", 0) / 1000 for r in runs),
                "cost_per_success_usd": (
                    total_cost / n_ok if (total_cost is not None and n_ok) else None
                ),
            }
        )
    return rows


def _fmt(key: str, value: Any) -> str:
    if value is None:
        return "-"
    if key == "success_rate":
        return f"{value * 100:.0f}"
    if key == "cost_per_success_usd":
        return f"${value:.4f}"
    if key == "med_secs":
        return f"{value:.1f}"
    if key in {"med_iters", "med_total_tokens", "med_thinking_tokens"}:
        return f"{value:g}"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    header = "  ".join(label.ljust(width) for _, label, width in _COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(_fmt(key, row[key]).ljust(width) for key, _, width in _COLUMNS))


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key, _, _ in _COLUMNS])
        writer.writeheader()
        writer.writerows({key: row[key] for key, _, _ in _COLUMNS} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_dir", nargs="?", default="logs", help="log directory (default logs/)")
    parser.add_argument("--csv", metavar="FILE", help="also write the table as CSV")
    args = parser.parse_args()

    summaries = load_summaries(args.log_dir)
    if not summaries:
        raise SystemExit(f"no summary records found in {args.log_dir}/*.jsonl")

    rows = aggregate(summaries)
    print(f"{len(summaries)} runs across {len(rows)} (problem x model x thinking) cells\n")
    print_table(rows)
    if args.csv:
        write_csv(args.csv, rows)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
