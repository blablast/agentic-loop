"""Cost computation from the (separate, offline) price table. Prices are explicit — an
unknown model raises rather than guessing."""

import pytest
from pricing import cost_of_summary, cost_usd


def test_cost_usd_combines_input_and_output_rates():
    # gemini-3.5-flash = $1.50 in / $9.00 out per 1M tokens
    assert cost_usd("gemini-3.5-flash", 1_000_000, 1_000_000) == pytest.approx(10.50)
    assert cost_usd("gemini-3.5-flash", 500_000, 0) == pytest.approx(0.75)


def test_cost_usd_unknown_model_raises():
    with pytest.raises(KeyError):
        cost_usd("gpt-9", 1000, 1000)


def test_cost_of_summary_counts_thinking_as_output():
    summary = {
        "model": "gemini-3.1-flash-lite",  # $0.25 in / $1.50 out
        "prompt_tokens_total": 1_000_000,
        "candidate_tokens_total": 400_000,
        "thoughts_tokens_total": 600_000,  # billed at the output rate, like candidates
    }
    # 0.25*1M + 1.50*(0.4M + 0.6M) = 0.25 + 1.50 = 1.75
    assert cost_of_summary(summary) == pytest.approx(1.75)
