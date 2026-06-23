"""GeminiMaker — a generic maker (works for any `Problem`) backed by Google Gemini.

It turns *any* `Problem` into candidates: render the problem's prompt, stream the answer,
and hand the raw text to `problem.parse_answer`. Nothing subset-sum-specific lives here —
not the prompt, not even the answer format — so the maker is reusable across problems.

Observability is opt-in and display-agnostic via `StreamHooks` (shared with the other
backends in `streaming.py`): the maker never imports a console or a color.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from .streaming import BaseStreamingMaker, Kind, StreamHooks, Usage

DEFAULT_MODEL = "gemini-3.5-flash"  # agentic, cheap; for max reasoning use gemini-3.1-pro


class GeminiMaker(BaseStreamingMaker):
    """Real maker backed by Google Gemini. Requires GEMINI_API_KEY (or GOOGLE_API_KEY).

        pip install google-genai
        export GEMINI_API_KEY=...

    thinking_level="high" helps on harder instances. The verifier's feedback flows
    into the next prompt (via `problem.render_prompt`) — that is what makes it agentic.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        thinking_level: str = "high",
        hooks: StreamHooks | None = None,
    ) -> None:
        from google import genai  # local import: deterministic demo works without the SDK
        from google.genai import types

        self._types = types
        self.client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        self.model = model
        self.thinking_level = thinking_level
        self.hooks = hooks or StreamHooks()

    def _config(self) -> Any:
        types = self._types
        level = self.thinking_level
        if isinstance(level, str):  # accept "high"/"low"/... and map to the SDK enum
            level = types.ThinkingLevel(level.upper())
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=level,
                include_thoughts=True,  # stream the reasoning parts, not only the answer
            ),
            # For Gemini 3.x do NOT change temperature/top_p (Google's guidance — it can
            # cause looping/worse reasoning). With thinking on, do not set a tiny
            # max_output_tokens either, or the visible answer may come back empty.
        )

    def _stream(self, prompt: str) -> str:
        """Stream the response, emit tokens to `on_token` and final usage to `on_usage`,
        return the full answer text."""
        answer: list[str] = []
        usage: Any = None
        start = time.perf_counter()  # cloud model is always warm; no local load to exclude
        chunks = self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=self._config(),
        )
        for chunk in chunks:
            for kind, text in self._tokens(chunk):
                if kind == "answer":
                    answer.append(text)
                if self.hooks.on_token:
                    self.hooks.on_token(kind, text)
            meta = getattr(chunk, "usage_metadata", None)  # last chunk carries cumulative counts
            if meta is not None:
                usage = meta
        latency_ms = (time.perf_counter() - start) * 1000.0
        if self.hooks.on_usage and usage is not None:
            self.hooks.on_usage(self._usage(usage, latency_ms))
        return "".join(answer)

    def _usage(self, meta: Any, latency_ms: float) -> Usage:
        # thoughts_token_count is None when thinking is off -> coerce to 0
        return Usage(
            prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            candidate_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            thoughts_tokens=getattr(meta, "thoughts_token_count", 0) or 0,
            total_tokens=getattr(meta, "total_token_count", 0) or 0,
            model=self.model,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _tokens(chunk: Any) -> Iterator[tuple[Kind, str]]:
        """Yield (kind, text) for each part of a chunk; kind is 'thinking' or 'answer'."""
        try:
            parts = chunk.candidates[0].content.parts or []
        except (AttributeError, IndexError, TypeError):
            return
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                yield ("thinking" if getattr(part, "thought", False) else "answer"), text
