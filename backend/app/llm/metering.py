"""Every model call goes through ``MeteredLLM``: it checks the daily budget before the call
and adds the call's cost to it afterwards, including calls whose answer was unusable."""

import logging
from collections.abc import Sequence

from app.limits import DailyBudget
from app.llm.base import (
    LLMClient,
    LLMError,
    LLMReply,
    Message,
    OutputT,
    ToolChoice,
    ToolSpec,
    Usage,
    cost_usd,
)

log = logging.getLogger("rag")


class MeteredLLM:
    def __init__(self, llm: LLMClient, budget: DailyBudget) -> None:
        self._llm = llm
        self._budget = budget

    @property
    def provider(self) -> str:
        return self._llm.provider

    @property
    def model(self) -> str:
        return self._llm.model

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
        self._budget.ensure_available()
        try:
            reply = self._llm.respond(
                system=system,
                messages=messages,
                tools=tools,
                output_type=output_type,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
            )
        except LLMError as error:
            if error.usage is not None:  # billed although unusable
                self._record(error.model or self.model, error.usage, output_type.__name__)
            raise
        self._record(reply.model, reply.usage, output_type.__name__)
        return reply

    def _record(self, model: str, usage: Usage, output: str) -> None:
        cost = cost_usd(model, usage)
        self._budget.add(cost)
        log.info(
            "model call model=%s output=%s tokens_in=%d tokens_out=%d cost_usd=%.5f",
            model,
            output,
            usage.input_tokens,
            usage.output_tokens,
            cost,
        )
