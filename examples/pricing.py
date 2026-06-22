"""Gemini API prices (USD per 1M tokens) for OFFLINE cost analysis. Kept out of the logs
on purpose — the logs store raw token counts only, because prices change. Update this one
table and recompute; never bake prices into the logs.

Source: https://ai.google.dev/gemini-api/docs/pricing — verified 2026-06-22.
Paid tier, standard (non-batch) API, prompts <=200K tokens. The output price already
*includes* thinking tokens, so cost uses (candidate + thoughts) as the output count.
"""

from __future__ import annotations

PRICES_AS_OF = "2026-06-22"
PRICES_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"

# model id -> (input $/1M, output $/1M).  Output covers candidate + thinking tokens.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-pro": (2.00, 12.00),  # <=200K context; 4.00 / 18.00 above 200K
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}


def cost_usd(model: str, prompt_tokens: int, output_tokens: int) -> float:
    """Cost of a run in USD. `output_tokens` = candidate + thinking tokens (Gemini bills
    thinking at the output rate). Raises KeyError for an unknown model — prices are
    explicit, never guessed."""
    input_price, output_price = PRICES[model]
    return (prompt_tokens * input_price + output_tokens * output_price) / 1_000_000


def cost_of_summary(summary: dict[str, object]) -> float:
    """Cost of one run from its `summary` JSONL record (token totals + model)."""
    model = str(summary["model"])
    prompt = int(summary["prompt_tokens_total"])  # type: ignore[arg-type]
    output = int(summary["candidate_tokens_total"]) + int(summary["thoughts_tokens_total"])  # type: ignore[arg-type]
    return cost_usd(model, prompt, output)
