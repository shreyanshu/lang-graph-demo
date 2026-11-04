"""Helpers shared by the tutorial scripts.

The important piece is ``ScriptedChatModel``: a fake chat model that replays a
fixed list of ``AIMessage`` objects. It lets every tutorial run **without an API
key**, while behaving like a real tool-calling model (it can emit ``tool_calls``).

Only file ``10_tools_and_prebuilt_agents.py`` talks to Gemini. Everything else
runs offline, so the lessons are free and deterministic.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


# ---------------------------------------------------------------------------
# fake model
# ---------------------------------------------------------------------------
class ScriptedChatModel(BaseChatModel):
    """Replays ``script`` in order, one AIMessage per call."""

    script: list[AIMessage]
    cursor: int = 0
    seen_tools: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        self.seen_tools = [getattr(t, "name", str(t)) for t in tools]
        return self

    def _next(self) -> AIMessage:
        if self.cursor < len(self.script):
            message = self.script[self.cursor]
            self.cursor += 1
            return message
        return AIMessage(content="[script exhausted]")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ):
        """Emit the scripted answer word by word (for stream_mode='messages')."""
        message = self._next()
        for word in str(message.content).split(" "):
            chunk = AIMessageChunk(content=word + " ")
            if run_manager:
                run_manager.on_llm_new_token(word + " ", chunk=chunk)
            yield ChatGenerationChunk(message=chunk)


def tool_call(name: str, args: dict, *, call_id: str | None = None) -> dict:
    """Build the ``tool_calls`` entry a real model would produce."""
    return {
        "name": name,
        "args": args,
        "id": call_id or f"call_{name}",
        "type": "tool_call",
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
def banner(text: str) -> None:
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def note(text: str) -> None:
    """Print an explanation between code sections."""
    import textwrap

    print(f"\n[what this shows]\n{textwrap.dedent(text).strip()}\n")


def show(label: str, value: Any) -> None:
    print(f"\n--- {label} " + "-" * max(0, 60 - len(label)))
    if isinstance(value, dict):
        for key, item in value.items():
            print(f"  {key}: {item!r}")
    else:
        print(f"  {value!r}")


def show_mermaid(graph) -> None:
    print("\n--- mermaid " + "-" * 50)
    print(graph.get_graph().draw_mermaid())
