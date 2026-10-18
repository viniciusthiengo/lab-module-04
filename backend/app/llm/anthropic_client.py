"""Anthropic (Claude) adapter of ``LLMClient``, using the official Python SDK."""

import json
from collections.abc import Sequence
from typing import Any, Literal

import anthropic
from anthropic import transform_schema
from pydantic import ValidationError

from app.llm.base import (
    LLMError,
    LLMReply,
    Message,
    OutputT,
    ToolCall,
    ToolChoice,
    ToolSpec,
    Usage,
    UseTool,
)

DEFAULT_MODEL = "claude-haiku-4-5"


class AnthropicClient:
    """Calls the Messages API with client tools and a JSON-schema output format.

    Tool inputs use strict schemas, and the final answer is constrained to the output type's
    JSON schema (structured outputs), then validated again with Pydantic.
    """

    provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        thinking: Literal["disabled", "adaptive"] | None = None,
        client: Any = None,
        timeout: float = 180.0,
    ) -> None:
        """``thinking`` is sent when set; otherwise each model uses its default (some think
        by default, and thinking tokens count toward ``max_tokens``)."""
        self._client = client or anthropic.Anthropic(
            api_key=api_key, timeout=timeout, max_retries=2
        )
        self._model = model
        self._thinking = thinking

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
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [_to_param(message) for message in messages],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": transform_schema(output_type.model_json_schema()),
                }
            },
        }
        if self._thinking:
            params["thinking"] = {"type": self._thinking}
        if tools:
            params["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": dict(tool.input_schema),
                    "strict": True,
                }
                for tool in tools
            ]
            params["tool_choice"] = (
                {"type": "tool", "name": tool_choice.name}
                if isinstance(tool_choice, UseTool)
                else {"type": tool_choice}
            )
        try:
            message = self._client.messages.create(**params)
        except anthropic.AuthenticationError as error:
            raise LLMError("The AI provider rejected the API key.") from error
        except anthropic.RateLimitError as error:
            raise LLMError(
                "The AI provider is busy (rate limited); try again in a minute.", retryable=True
            ) from error
        except anthropic.APIConnectionError as error:
            raise LLMError("Could not reach the AI provider.", retryable=True) from error
        except anthropic.APIStatusError as error:
            retryable = error.status_code >= 500
            raise LLMError(
                f"The AI provider failed (HTTP {error.status_code}).", retryable=retryable
            ) from error
        return self._to_reply(message, output_type)

    def _to_reply(self, message: Any, output_type: type[OutputT]) -> LLMReply:
        usage = Usage(
            input_tokens=message.usage.input_tokens or 0,
            output_tokens=message.usage.output_tokens or 0,
            cache_read_tokens=message.usage.cache_read_input_tokens or 0,
            cache_write_tokens=message.usage.cache_creation_input_tokens or 0,
        )
        tool_calls = tuple(
            ToolCall(block.id, block.name, dict(block.input))
            for block in message.content
            if block.type == "tool_use"
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        reason = message.stop_reason
        billed = {"usage": usage, "model": message.model}
        if reason == "refusal":
            raise LLMError("The model declined this request.", **billed)
        if reason == "max_tokens":
            raise LLMError("The model's answer hit the output limit; try smaller files.", **billed)
        output = None
        if not tool_calls:
            try:
                output = output_type.model_validate_json(text)
            except ValidationError as error:
                raise LLMError(
                    "The model returned an answer in an unexpected format.", **billed
                ) from error
        return LLMReply(tool_calls, output, reason or "end_turn", usage, message.model, text)


def _to_param(message: Message) -> dict[str, Any]:
    """Converts a neutral message to the Messages API format."""
    if message.tool_results:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": result.call_id,
                    "content": result.content
                    if isinstance(result.content, str)
                    else json.dumps(result.content),
                    "is_error": result.is_error,
                }
                for result in message.tool_results
            ],
        }
    if message.tool_calls:
        content: list[dict[str, Any]] = []
        if message.text:
            content.append({"type": "text", "text": message.text})
        content += [
            {"type": "tool_use", "id": call.id, "name": call.name, "input": dict(call.arguments)}
            for call in message.tool_calls
        ]
        return {"role": "assistant", "content": content}
    return {"role": message.role, "content": message.text}
