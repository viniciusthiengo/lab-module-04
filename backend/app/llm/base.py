"""Provider-neutral types of the LLM layer.

The agent only talks to ``LLMClient``; each provider adapter converts these types to its own
API. Adding a provider means writing one adapter.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call, described by a JSON schema."""

    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the model."""

    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The answer to a tool call (any JSON-serializable content)."""

    call_id: str
    content: Any
    is_error: bool = False


@dataclass(frozen=True)
class Message:
    """One turn of the conversation: user text, or an assistant turn with tool calls, or the
    results of those calls (sent back as a user turn)."""

    role: Literal["user", "assistant"]
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()


@dataclass(frozen=True)
class Usage:
    """Tokens of one or more calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True)
class LLMReply:
    """One model answer: tool calls to run, or the final structured output."""

    tool_calls: tuple[ToolCall, ...]
    output: BaseModel | None
    stop_reason: str
    usage: Usage
    model: str
    text: str = ""


@dataclass(frozen=True)
class UseTool:
    """Makes the model call the tool ``name`` (it may call it several times)."""

    name: str


ToolChoice = Literal["auto", "none"] | UseTool
"""``auto``: the model calls tools or answers; ``none``: it must answer; ``UseTool``: it must
call that tool."""


class LLMError(Exception):
    """A provider failure, with a message that can be shown to the user.

    ``usage`` and ``model`` are set when the model did answer (and the call was billed) but the
    answer is unusable: cut by the output limit, refused, or in the wrong format.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        usage: Usage | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.usage = usage
        self.model = model


class LLMClient(Protocol):
    """What the agent needs from a provider."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

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
        """One model call. When the model answers without tools, ``output`` holds an
        ``output_type`` instance; ``tool_choice="none"`` forces that final answer."""
        ...


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""

    input: float
    output: float


PRICES: Mapping[str, Price] = {
    "claude-haiku-4-5": Price(1.0, 5.0),
    "claude-sonnet-5": Price(2.0, 10.0),
    "claude-opus-5": Price(5.0, 25.0),
}
_MOST_EXPENSIVE = max(PRICES.values(), key=lambda price: price.output)


def cost_usd(model: str, usage: Usage) -> float:
    """Cost of ``usage`` on ``model`` (unknown models are priced as the most expensive).

    Cache writes cost 1.25x and cache reads 0.1x the input price.
    """
    price = next((value for key, value in PRICES.items() if model.startswith(key)), _MOST_EXPENSIVE)
    return (
        usage.input_tokens * price.input
        + usage.cache_write_tokens * price.input * 1.25
        + usage.cache_read_tokens * price.input * 0.1
        + usage.output_tokens * price.output
    ) / 1_000_000
