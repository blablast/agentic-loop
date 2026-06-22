"""OllamaMaker — a second problem-agnostic maker, backed by a local Ollama server.

Same contract and seams as `GeminiMaker`: render the problem's prompt, stream the reply,
hand the raw text to `problem.parse_answer`, and emit tokens + a final `Usage` to the
injected `StreamHooks`. So the loop, the problems, and the logger don't change at all —
only the backend behind the maker does.

Talks to Ollama's `/api/chat` over plain HTTP with the standard library (no extra
dependency, mirroring how the Gemini SDK import is kept optional). Thinking-capable
models stream their reasoning in a separate `message.thinking` field, which maps onto the
"thinking" token kind; the visible reply is `message.content`.

Token note: Ollama reports `prompt_eval_count` (input) and `eval_count` (all generated
tokens, reasoning included). It does NOT split a separate thinking-token count, so `Usage`
records `candidate_tokens = eval_count` and `thoughts_tokens = 0`. Local generation is
free, so $/success is a Gemini-only metric — locally compare tokens, iterations, success.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from .streaming import BaseStreamingMaker, StreamHooks, Usage

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class OllamaMaker(BaseStreamingMaker):
    """Generic maker backed by a local Ollama server (`ollama serve`).

        maker = OllamaMaker(model="gemma4:31b", host="http://localhost:11434")

    `think` toggles a model's reasoning channel: True/False, or a level string like
    "high" for models that accept one (e.g. gpt-oss). The verifier's feedback flows into
    the next prompt via `problem.render_prompt` — same agentic loop as Gemini.
    """

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_OLLAMA_HOST,
        think: bool | str | None = True,
        hooks: StreamHooks | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.url = host.rstrip("/") + "/api/chat"
        self.think = think
        self.hooks = hooks or StreamHooks()
        self.timeout = timeout

    def _stream(self, prompt: str) -> str:
        """Stream the chat reply, emit tokens to `on_token` and final usage to `on_usage`,
        return the full visible answer text."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        if self.think is not None:
            body["think"] = self.think
        request = urllib.request.Request(  # noqa: S310 - fixed http(s) Ollama endpoint
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        answer: list[str] = []
        prompt_tokens = eval_tokens = 0
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            for line in response:  # Ollama streams newline-delimited JSON objects
                raw = line.strip()
                if not raw:
                    continue
                chunk = json.loads(raw)
                message = chunk.get("message") or {}
                thinking = message.get("thinking")
                content = message.get("content")
                if thinking and self.hooks.on_token:
                    self.hooks.on_token("thinking", thinking)
                if content:
                    answer.append(content)
                    if self.hooks.on_token:
                        self.hooks.on_token("answer", content)
                if chunk.get("done"):
                    prompt_tokens = chunk.get("prompt_eval_count") or 0
                    eval_tokens = chunk.get("eval_count") or 0

        if self.hooks.on_usage:
            self.hooks.on_usage(
                Usage(
                    prompt_tokens=prompt_tokens,
                    candidate_tokens=eval_tokens,  # all generated tokens (reasoning included)
                    thoughts_tokens=0,  # Ollama does not split a separate thinking count
                    total_tokens=prompt_tokens + eval_tokens,
                    model=self.model,
                )
            )
        return "".join(answer)
