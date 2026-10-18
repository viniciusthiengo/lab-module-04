"""Offline ``LLMClient`` for tests: a function decides each reply.

The agent cannot tell it from a real provider: tool calls, structured outputs and usage flow
through the same types. Every request is recorded for assertions.
"""

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.llm.base import (
    LLMReply,
    Message,
    OutputT,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolSpec,
    Usage,
    UseTool,
)

FAKE_MODEL = "fake-model"
FAKE_USAGE = Usage(1000, 400)


@dataclass
class ScriptedTurn:
    """A canned reply: tool calls to request, or the final output."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    output: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class FakeRequest:
    """What the agent sent in one call."""

    system: str
    messages: tuple[Message, ...]
    tools: tuple[str, ...]
    output_type: type[BaseModel]
    tool_choice: ToolChoice
    max_tokens: int

    @property
    def prompt(self) -> str:
        """The first user message: the task."""
        return self.messages[0].text

    @property
    def tool_results(self) -> list[ToolResult]:
        """Every tool result sent back so far, oldest first."""
        return [result for message in self.messages for result in message.tool_results]


Responder = Callable[[FakeRequest], ScriptedTurn]


class FakeLLMClient:
    """Replies with whatever ``responder`` returns for each request.

    A responder may also raise ``LLMError`` to simulate a provider failure. Calls may come from
    several threads at once (parallel steps), so the request log is locked.
    """

    provider = "fake"

    def __init__(
        self, responder: Responder, model: str = FAKE_MODEL, usage: Usage = FAKE_USAGE
    ) -> None:
        self._responder = responder
        self._model = model
        self._usage = usage
        self._lock = threading.Lock()
        self.requests: list[FakeRequest] = []

    @property
    def model(self) -> str:
        return self._model

    def respond(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        output_type: type[OutputT],
        tool_choice: ToolChoice = "auto",
        max_tokens: int = 4096,
    ) -> LLMReply:
        request = FakeRequest(
            system,
            tuple(messages),
            tuple(tool.name for tool in tools),
            output_type,
            tool_choice,
            max_tokens,
        )
        with self._lock:
            self.requests.append(request)
        turn = self._responder(request)
        # Like the real API: a forced tool is always called.
        forced = tool_choice.name if isinstance(tool_choice, UseTool) else None
        if forced and not any(call.name == forced for call in turn.tool_calls):
            raise AssertionError(f"the scripted turn must call {forced}")
        if turn.tool_calls and tool_choice != "none":
            return LLMReply(tuple(turn.tool_calls), None, "tool_use", self._usage, self._model)
        output = output_type.model_validate(turn.output or {})
        text = json.dumps(turn.output or {})
        return LLMReply((), output, "end_turn", self._usage, self._model, text)

    def calls_for(self, output_type: type[BaseModel]) -> list[FakeRequest]:
        """The requests that asked for ``output_type``."""
        with self._lock:
            return [request for request in self.requests if request.output_type is output_type]
