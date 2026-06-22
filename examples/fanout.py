"""Tee several observers into one. The console and the session logger are independent
observers of the same events; the runner fans them into a single `LoopView` / `StreamHooks`
so the core still sees exactly one of each (no new seam in the core)."""

from __future__ import annotations

from agentic_loop import LoopView, StreamHooks


def tee_view(*views: LoopView) -> LoopView:
    def fan(method: str, *args: object) -> None:
        for view in views:
            callback = getattr(view, method)
            if callback is not None:
                callback(*args)

    return LoopView(
        on_attempt=lambda i: fan("on_attempt", i),
        on_verdict=lambda i, candidate, ok, feedback: fan("on_verdict", i, candidate, ok, feedback),
    )


def tee_hooks(*hooks: StreamHooks) -> StreamHooks:
    def fan(method: str, *args: object) -> None:
        for hook in hooks:
            callback = getattr(hook, method)
            if callback is not None:
                callback(*args)

    return StreamHooks(
        on_prompt=lambda text: fan("on_prompt", text),
        on_token=lambda kind, text: fan("on_token", kind, text),
        on_usage=lambda usage: fan("on_usage", usage),
    )
