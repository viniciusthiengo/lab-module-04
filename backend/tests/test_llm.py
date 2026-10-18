import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from app.limits import BudgetExceededError, DailyBudget
from app.llm.anthropic_client import AnthropicClient
from app.llm.base import LLMError, Message, ToolCall, ToolSpec, Usage, UseTool, cost_usd
from app.llm.fake import FakeLLMClient, FakeRequest, ScriptedTurn
from app.llm.metering import MeteredLLM
from app.outputs import Answer
from tests.fakes import ScriptedClaude

ANSWER = {"answer": "It hashes them [1].", "sources_used": [1], "grounded": True}
LOOKUP = ToolSpec("lookup", "Looks a symbol up.", {"type": "object", "properties": {}})


def _message(content: list[Any], stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        model="claude-haiku-4-5-20251001",
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=200,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=0,
        ),
    )


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


class FakeSDK:
    """Stands in for ``anthropic.Anthropic``: records the parameters, returns ``reply`` or
    raises ``error``."""

    def __init__(self, reply: SimpleNamespace | None = None, error: Exception | None = None):
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)
        self._reply = reply
        self._error = error

    def _create(self, **params: Any) -> SimpleNamespace | None:
        self.calls.append(params)
        if self._error is not None:
            raise self._error
        return self._reply


def _status_error(kind: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return kind("error", response=httpx.Response(status, request=request), body=None)


def respond(client: AnthropicClient, **overrides: Any):
    params: dict[str, Any] = {
        "system": "sys",
        "messages": [Message("user", text="Question?")],
        "tools": [],
        "output_type": Answer,
        **overrides,
    }
    return client.respond(**params)


class TestAnthropicAdapter:
    def test_sends_the_output_schema_and_parses_the_answer(self) -> None:
        sdk = FakeSDK(_message([_text(json.dumps(ANSWER))]))

        reply = respond(AnthropicClient("test-key", client=sdk), max_tokens=2_048)

        params = sdk.calls[0]
        assert (params["model"], params["max_tokens"], params["system"]) == (
            "claude-haiku-4-5",
            2_048,
            "sys",
        )
        assert params["messages"] == [{"role": "user", "content": "Question?"}]
        schema = params["output_config"]["format"]["schema"]
        assert set(schema["properties"]) == {"answer", "sources_used", "grounded"}
        assert schema["additionalProperties"] is False
        assert "tools" not in params
        assert reply.output == Answer(**ANSWER)
        assert reply.usage == Usage(1000, 200, 0, 0)
        assert reply.model == "claude-haiku-4-5-20251001"

    def test_strict_tools_and_tool_choice(self) -> None:
        tool_use = SimpleNamespace(type="tool_use", id="t1", name="lookup", input={"q": "x"})
        sdk = FakeSDK(_message([_text("Looking."), tool_use], "tool_use"))

        reply = respond(
            AnthropicClient("k", client=sdk), tools=[LOOKUP], tool_choice=UseTool("lookup")
        )

        params = sdk.calls[0]
        assert params["tools"][0]["strict"] is True
        assert params["tool_choice"] == {"type": "tool", "name": "lookup"}
        assert reply.tool_calls == (ToolCall("t1", "lookup", {"q": "x"}),)
        assert reply.output is None

    def test_thinking_is_sent_only_when_configured(self) -> None:
        reply = _message([_text(json.dumps(ANSWER))])
        default, disabled = FakeSDK(reply), FakeSDK(reply)

        respond(AnthropicClient("k", client=default))
        respond(AnthropicClient("k", thinking="disabled", client=disabled))

        assert "thinking" not in default.calls[0]
        assert disabled.calls[0]["thinking"] == {"type": "disabled"}

    @pytest.mark.parametrize(
        ("content", "stop_reason", "message"),
        [
            ([_text("{}")], "refusal", "declined"),
            ([_text('{"answer": "trunc')], "max_tokens", "output limit"),
            ([_text("not json")], "end_turn", "unexpected format"),
        ],
    )
    def test_unusable_answers_are_billed_errors(
        self, content: list[Any], stop_reason: str, message: str
    ) -> None:
        client = AnthropicClient("k", client=FakeSDK(_message(content, stop_reason)))

        with pytest.raises(LLMError, match=message) as error:
            respond(client)
        assert error.value.usage == Usage(1000, 200, 0, 0)
        assert error.value.model == "claude-haiku-4-5-20251001"

    @pytest.mark.parametrize(
        ("error", "message", "retryable"),
        [
            (_status_error(anthropic.AuthenticationError, 401), "rejected the API key", False),
            (_status_error(anthropic.RateLimitError, 429), "busy", True),
            (_status_error(anthropic.InternalServerError, 500), "HTTP 500", True),
            (_status_error(anthropic.BadRequestError, 400), "HTTP 400", False),
            (
                anthropic.APIConnectionError(
                    request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                ),
                "Could not reach",
                True,
            ),
        ],
    )
    def test_provider_errors_become_clear_messages(
        self, error: Exception, message: str, retryable: bool
    ) -> None:
        client = AnthropicClient("k", client=FakeSDK(error=error))

        with pytest.raises(LLMError, match=message) as raised:
            respond(client)
        assert raised.value.retryable is retryable
        assert raised.value.usage is None


def test_costs_follow_the_model_prices() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000)

    assert cost_usd("claude-haiku-4-5", usage) == pytest.approx(1.5)
    assert cost_usd("claude-haiku-4-5-20251001", usage) == pytest.approx(1.5)
    assert cost_usd("claude-sonnet-5", usage) == pytest.approx(3.0)
    assert cost_usd("unknown-model", usage) == pytest.approx(7.5)  # priced as the most expensive
    cached = Usage(cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
    assert cost_usd("claude-haiku-4-5", cached) == pytest.approx(0.1 + 1.25)


class TestMeteredLLM:
    def test_every_call_is_added_to_the_daily_budget(self) -> None:
        budget = DailyBudget(1.0)
        answering = FakeLLMClient(lambda _: ScriptedTurn(output=ANSWER), model="claude-haiku-4-5")
        metered = MeteredLLM(answering, budget)

        metered.respond(
            system="s", messages=[Message("user", text="Q")], tools=[], output_type=Answer
        )

        assert budget.spent_usd == pytest.approx((1000 * 1 + 400 * 5) / 1e6)
        assert metered.model == "claude-haiku-4-5"

    def test_billed_failures_are_counted_too(self) -> None:
        def refused(_: FakeRequest) -> ScriptedTurn:
            raise LLMError("declined", usage=Usage(1000, 0), model="claude-haiku-4-5")

        budget = DailyBudget(1.0)
        metered = MeteredLLM(FakeLLMClient(refused), budget)

        with pytest.raises(LLMError):
            metered.respond(system="s", messages=[], tools=[], output_type=Answer)

        assert budget.spent_usd == pytest.approx(0.001)

    def test_no_call_once_the_budget_is_spent(self) -> None:
        llm = FakeLLMClient(ScriptedClaude())
        budget = DailyBudget(0.01)
        budget.add(0.01)

        with pytest.raises(BudgetExceededError):
            MeteredLLM(llm, budget).respond(system="s", messages=[], tools=[], output_type=Answer)

        assert llm.requests == []
