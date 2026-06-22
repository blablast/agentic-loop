"""Agentic loop: a maker proposes, an external verifier decides.

The point is the generator–verifier asymmetry — the gate that decides success is
external to the maker, because a model's self-assessment is unreliable.

This top-level package is the **generic, problem-agnostic core**:
  contracts - the abstractions (`Problem`, `Maker`, `Verifier`) everything depends on
  loop      - the harness (STATE / EXECUTE / VERIFY / ITERATE / STOP) + `LoopView`
  streaming - shared maker observability (`StreamHooks`, `Usage`)
  gemini    - `GeminiMaker`, a generic maker for any problem (cloud)
  ollama    - `OllamaMaker`, the same contract against a local Ollama server

Concrete problems live in the sibling `problems` package (e.g. `problems.subset_sum`);
the core never imports them, so adding a problem touches no code here.
"""

from .contracts import Candidate, History, Maker, Problem, Verifier
from .gemini import DEFAULT_MODEL, GeminiMaker
from .loop import LoopView, loop
from .ollama import DEFAULT_OLLAMA_HOST, OllamaMaker
from .streaming import StreamHooks, Usage

__all__ = [
    "loop",
    "LoopView",
    "Problem",
    "Maker",
    "Verifier",
    "Candidate",
    "History",
    "GeminiMaker",
    "OllamaMaker",
    "StreamHooks",
    "Usage",
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_HOST",
]
