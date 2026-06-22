"""SessionLogger writes correct JSONL from faked events — no network, no SDK. The logger
is the observation layer wired through the same seams as the console.
"""

import json

from session_log import RunConfig, SessionLogger

from agentic_loop import Usage


def _config():
    return RunConfig(
        problem="subset_sum",
        model="gemini-3.5-flash",
        thinking_level="high",
        seed=7,
        max_iters=12,
        sdk_version="1.2.3",
    )


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_logger_writes_attempt_and_summary(tmp_path):
    path = tmp_path / "out.jsonl"
    logger = SessionLogger(str(path), "run123", _config(), clock=lambda: 0.0)
    logger.on_attempt(1)
    logger.on_prompt("PROMPT 1")
    logger.on_usage(Usage(100, 20, 50, 170, "gemini-3.5-flash"))
    logger.on_verdict(1, [1, 2], False, "różnica +3")  # a batch of two candidates...
    logger.on_verdict(1, [3, 4], True, "OK")  # ...the second solves
    logger.close()

    records = _records(path)
    attempts = [r for r in records if r["record_type"] == "attempt"]
    summary = next(r for r in records if r["record_type"] == "summary")

    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt["schema_version"] == 1
    assert attempt["run_id"] == "run123"
    assert attempt["iter"] == 1
    assert attempt["model"] == "gemini-3.5-flash"
    assert (attempt["prompt_tokens"], attempt["thoughts_tokens"], attempt["total_tokens"]) == (
        100,
        50,
        170,
    )
    assert attempt["solved_here"] is True
    assert len(attempt["candidates"]) == 2 and attempt["candidates"][1]["ok"] is True

    assert summary["solved"] is True
    assert summary["iters"] == 1
    assert summary["total_tokens_total"] == 170
    assert summary["thoughts_tokens_total"] == 50
    assert summary["config"]["sdk_version"] == "1.2.3"


def test_logger_iteration_without_usage_logs_zeros(tmp_path):
    path = tmp_path / "out.jsonl"
    with SessionLogger(str(path), "r", _config(), clock=lambda: 0.0) as logger:
        logger.on_attempt(1)
        logger.on_verdict(1, [1], False, "x")  # no on_usage this turn

    attempt = next(r for r in _records(path) if r["record_type"] == "attempt")
    assert attempt["total_tokens"] == 0 and attempt["solved_here"] is False


def test_logger_writes_transcripts_to_second_file(tmp_path):
    metrics = tmp_path / "m.jsonl"
    transcripts = tmp_path / "m.transcripts.jsonl"
    with SessionLogger(
        str(metrics), "r", _config(), transcript_path=str(transcripts), clock=lambda: 0.0
    ) as logger:
        logger.on_attempt(1)
        logger.on_prompt("ASK")
        logger.on_token("thinking", "hmm ")
        logger.on_token("answer", "[1, 2]")
        logger.on_usage(Usage(1, 2, 3, 4, "m"))
        logger.on_verdict(1, [1, 2], True, "OK")

    transcript = _records(transcripts)[0]
    assert transcript["record_type"] == "transcript"
    assert transcript["prompt"] == "ASK"
    assert transcript["thinking"] == "hmm " and transcript["answer"] == "[1, 2]"
    # the light metrics file stays free of transcript text
    assert all(r["record_type"] != "transcript" for r in _records(metrics))


def test_summary_includes_cost_with_price_snapshot(tmp_path):
    path = tmp_path / "out.jsonl"
    prices = {"gemini-3.5-flash": (1.50, 9.00)}  # config()'s model
    with SessionLogger(
        str(path), "r", _config(), prices=prices, prices_as_of="2026-06-22", clock=lambda: 0.0
    ) as logger:
        logger.on_attempt(1)
        logger.on_usage(Usage(1_000_000, 400_000, 600_000, 2_000_000, "gemini-3.5-flash"))
        logger.on_verdict(1, [1], True, "OK")

    summary = next(r for r in _records(path) if r["record_type"] == "summary")
    # 1.50*1M + 9.00*(0.4M + 0.6M) = 1.50 + 9.00 = 10.50 ; thinking billed at output rate
    assert summary["cost_usd"] == 10.5
    assert summary["cost_prices"] == {
        "as_of": "2026-06-22",
        "input_per_1m": 1.5,
        "output_per_1m": 9.0,
    }


def test_summary_cost_is_null_for_unknown_model(tmp_path):
    path = tmp_path / "out.jsonl"
    with SessionLogger(
        str(path), "r", _config(), prices={"other-model": (1.0, 1.0)}, clock=lambda: 0.0
    ) as logger:
        logger.on_attempt(1)
        logger.on_usage(Usage(100, 20, 50, 170, "gemini-3.5-flash"))
        logger.on_verdict(1, [1], True, "OK")

    summary = next(r for r in _records(path) if r["record_type"] == "summary")
    assert summary["cost_usd"] is None and summary["cost_prices"] is None
