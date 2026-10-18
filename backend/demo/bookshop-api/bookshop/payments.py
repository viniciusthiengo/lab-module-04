"""Card payments through the payment gateway, with retries."""

import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 0.5


class GatewayTimeout(Exception):
    """The gateway did not answer in time: the charge may be retried."""


class PaymentDeclined(Exception):
    """The card was declined: retrying would not help."""


@dataclass(frozen=True)
class Charge:
    id: str
    amount: Decimal
    status: str


class PaymentGateway(Protocol):
    def charge(self, card_token: str, amount: Decimal, idempotency_key: str) -> Charge: ...

    def refund(self, charge_id: str) -> None: ...


def charge_order(
    gateway: PaymentGateway, order_id: str, card_token: str, amount: Decimal, sleep=time.sleep
) -> Charge:
    """Charges the card, retrying timeouts up to MAX_ATTEMPTS times with exponential
    backoff (0.5 s, then 1 s). Declined cards are not retried. The order id is the
    idempotency key, so a retried charge is never billed twice."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return gateway.charge(card_token, amount, idempotency_key=order_id)
        except GatewayTimeout:
            if attempt == MAX_ATTEMPTS:
                raise
            delay = BACKOFF_SECONDS * 2 ** (attempt - 1)
            log.warning("gateway timeout for order %s, retrying in %.1fs", order_id, delay)
            sleep(delay)
    raise AssertionError("unreachable")
