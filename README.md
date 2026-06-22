# agentic-loop

A minimal, working example of an **agentic loop**: a model proposes, an external verifier
decides. The point is the **generator–verifier asymmetry** – the gate that decides success
must be *external* to the model, because a model's self-assessment is unreliable.

It ships five problems (each "easy to verify, hard to find"), two interchangeable backends
(Google Gemini in the cloud, any local Ollama model), per-run token/cost logging, and a
benchmark that sweeps a grid of model × thinking-level × problem × seed to study one
practical question: **for an agentic loop, is it better to pay for a stronger model, or
run a cheaper one for more attempts behind a solid gate?**

## The idea

`agentic_loop/loop.py` runs four phases until a candidate passes or a hard iteration limit:

```python
for candidate in maker(problem, history):       # EXECUTE: the maker proposes a batch
    ok, feedback = verify(problem, candidate)    # VERIFY: an external gate decides
    if ok:
        return candidate, i                      # STOP: success
    history.append((candidate, feedback))        # ITERATE: remember the failure, retry
```

- **STATE** – `history`, the `(candidate, feedback)` pairs from prior attempts.
- **EXECUTE** – a maker proposes a *batch* of candidates. Verification is cheap, so the
  loop checks every one and takes the first that passes (free best-of-N when the model
  rambles; a no-op when it returns a single clean answer).
- **VERIFY** – `verify(problem, candidate)` returns concrete feedback (e.g. `różnica +5`)
  that flows into the next prompt. That verbal signal is what makes it agentic, not blind retry.
- **STOP** – success, or `max_iters`.

The maker is swappable, the loop is identical: success is decided by the gate, not by the
author of the solution.

**No MCP, by design.** The verifier is not a tool the model calls; it is an external test
in the loop's code. Keeping the maker and the checker separate is the whole point.

## Layout

Three layers, each depending only on the ones below it (`agentic_loop` ← `problems` ← `examples`):

```
src/agentic_loop/   problem-agnostic core: contracts, loop, streaming hooks,
                    GeminiMaker (gemini.py), OllamaMaker (ollama.py)
src/problems/       five concrete problems + a REGISTRY (pick one by name)
examples/           entry points, presentation, and the benchmark (console, run,
                    sweep, analyze_logs, session logging, pricing)
tests/              the trusted gates + a shared contract over every registered problem
```

A `Problem` owns both ends of the model conversation: `render_prompt(history)` and
`parse_answer(text) -> list[candidate]`. So the makers stay fully generic – they never
know the answer format. A `candidate` is opaque to the core (`Candidate = Any`), so adding
a problem is one new file plus one `REGISTRY` line; the core and runners are untouched.

## The problems

| Problem | Candidate | What the model never sees |
|---|---|---|
| `subset_sum` | list of ints | nothing (fully observable; the NP "easy verify, hard find" case) |
| `optimization` | `[x, y]` point | the hidden anisotropic bowl; it only gets the loss per probe |
| `regex_golf` | regex string | a large hidden word set (only a few words are shown) |
| `reverse_engineering` | list of strings | the hidden input→output rule and the expected outputs |
| `maze` | list of moves | the hidden walls, mapped by probing |

Instances are randomized per seed (so the model can't memorize them) and every gate is
deterministic. Each problem's `describe()` records its full scale (including the hidden
parts) to the log, so an analyst can read the instance's difficulty next to the cost.

## Install

```bash
git clone https://github.com/<you>/agentic-loop.git
cd agentic-loop
uv sync          # creates the venv, installs the package (editable) + dev deps
```

[uv](https://docs.astral.sh/uv/) is the easiest path. Without it:

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .                                     # runtime deps
pip install pytest ruff mypy                         # dev tools, to run the tests/linters
```

Requires Python >= 3.13.

## Quick start (no API key)

```bash
uv run python examples/run_demo.py     # deterministic demo, no API key needed
uv run pytest -q                       # tests
```

The demo solves subset-sum with a brute-force maker and a feedback-using heuristic, and
shows the gate rejecting a confident-but-wrong answer. No model, no network.

## Run one problem with a model

Each iteration shows the exact prompt sent, the model's thinking and answer streaming live,
then the verifier's verdict. Token usage (thinking included) and cost are logged to `logs/`.
Pick a problem with `--problem` (`subset_sum`, `optimization`, `regex_golf`,
`reverse_engineering`, `maze`).

### Gemini (cloud)

Get a key at <https://aistudio.google.com/apikey>, then:

```bash
cp .env.example .env        # set GEMINI_API_KEY=... (or GOOGLE_API_KEY); .env is git-ignored
uv run python examples/run.py --problem maze --thinking high
```

`gemini-3.5-flash` (default) is fast and cheap, a good start for a loop that makes many
calls; `gemini-3.1-pro` is the strongest reasoning fallback (pass it with `--model`). Do
**not** change `temperature`/`top_p` for Gemini 3.x (Google's guidance; it can cause
looping and worse reasoning), and don't set a tiny `max_output_tokens` with thinking on, or
the visible answer can come back empty.

### Ollama (local, free)

Install Ollama, pull a model, and make sure the server is reachable:

```bash
curl -fsSL https://ollama.com/install.sh | sh    # Linux; on macOS install the app
ollama pull qwen3:4b                              # any model; reasoning models have a "thinking" capability
ollama list                                       # see what you have
curl http://localhost:11434/api/tags              # the server runs on :11434 by default
```

```bash
uv run python examples/run.py --backend ollama --model qwen3:4b \
  --ollama-host http://localhost:11434 --problem maze
```

For Ollama the thinking axis is on/off (`--thinking minimal` = off, anything else = on).
Local generation is free, so a dollar cost is reported only for Gemini. `--ollama-host` can
point at a remote server (for example over Tailscale).

Main flags (both backends): `--problem`, `--backend {gemini,ollama}`, `--model`,
`--thinking {minimal,low,medium,high}`, `-n` (instance size), `--max-iters`, `--seed`,
`--log-dir`, `--log-transcripts`, `--nolog`. See `--help` for all.

## The benchmark (stronger model vs. more attempts)

`sweep.py` runs the problems across a grid of models × thinking × seeds, one logged run per
cell, **using the same seeds for every cell** (a paired comparison – every model faces the
identical instances). `analyze_logs.py` turns the logs into a table of success rate, median
iterations, median tokens (total and thinking), and $/success.

```bash
# 1. edit CELLS at the top of examples/sweep.py to list your models
uv run python examples/sweep.py --dry-run                  # preview the plan
# 2. a pilot on the discriminating problems, into a fresh dir
uv run python examples/sweep.py \
    --problems subset_sum,optimization,maze --seeds 0-4 --log-dir logs_pilot \
    --ollama-host http://localhost:11434
# 3. read the table
uv run python examples/analyze_logs.py logs_pilot
```

The sweep is **resumable**: re-running skips cells whose log already has a `summary` record,
so an interrupted overnight run continues where it stopped (`--no-resume` forces a fresh
run). Token counts are logged raw; cost is computed offline from a price table
(`examples/pricing.py`), so a price change touches one file and old runs stay reproducible.

## What else you can do

- **Add a problem.** Implement a `Problem` (`render_prompt` + `parse_answer`), a `verify`,
  a `generate(n, seed)`, and a `describe()` in a new `src/problems/<name>.py`, then add one
  line to `REGISTRY`. The runners, the benchmark, and the shared test contract pick it up by
  name; the core is untouched.
- **Add a backend.** Subclass `BaseStreamingMaker` and implement `_stream`; the loop depends
  on the `Maker` protocol, not on a base class, so it plugs straight in.
- **Run the full benchmark.** List your models in `examples/sweep.py` (`CELLS`), add Gemini
  cells for real $/success, and sweep every problem over `--seeds 0-9`.
- **Tune difficulty.** `-n` controls subset-sum size; `_TOLERANCE` in `optimization.py` and
  `_WALL_DENSITY` in `maze.py` set their hardness, to land outcomes in the informative middle.

## Conventions

- Dependency direction is one-way: `agentic_loop` (core) ← `problems` ← `examples`. The
  core never imports a problem, a console, or a logger; display and logging are injected
  via callbacks (`LoopView`, `StreamHooks`).
- Each gate is deterministic and tested – it is the one component that must be incorruptible,
  so it is the one tested hardest.
- Requires Python >= 3.13. CI runs ruff + mypy (strict, over `src/`) + pytest.
