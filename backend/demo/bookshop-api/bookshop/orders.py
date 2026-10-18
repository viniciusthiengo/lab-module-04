"""Placing and cancelling orders."""

import uuid
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from bookshop.cart import Cart
from bookshop.inventory import Inventory
from bookshop.notifications import Notifier
from bookshop.payments import PaymentGateway, charge_order


class OrderStatus(Enum):
    PENDING = "pending"
    PAID = "paid"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"


class OrderError(Exception):
    """The order cannot be placed or changed."""


@dataclass
class Order:
    id: str
    user_email: str
    total: Decimal
    status: OrderStatus
    charge_id: str | None = None


def place_order(
    cart: Cart,
    user_email: str,
    card_token: str,
    inventory: Inventory,
    gateway: PaymentGateway,
    notifier: Notifier,
) -> Order:
    """Reserves the stock, charges the card and confirms by e-mail. If the payment fails,
    the reservation is released so the copies can be sold again."""
    if not cart.items:
        raise OrderError("the cart is empty")
    order = Order(str(uuid.uuid4()), user_email, cart.total(), OrderStatus.PENDING)
    inventory.reserve(order.id, {book_id: qty for book_id, (_, qty) in cart.items.items()})
    try:
        charge = charge_order(gateway, order.id, card_token, order.total)
    except Exception:
        inventory.release(order.id)
        raise
    order.status, order.charge_id = OrderStatus.PAID, charge.id
    notifier.order_confirmation(order)
    return order


def cancel_order(order: Order, inventory: Inventory, gateway: PaymentGateway) -> None:
    """Cancels an order that has not shipped: refunds the charge and releases the stock."""
    if order.status is OrderStatus.SHIPPED:
        raise OrderError("shipped orders cannot be cancelled; request a return instead")
    if order.status is OrderStatus.CANCELLED:
        return
    if order.charge_id is not None:
        gateway.refund(order.charge_id)
    inventory.release(order.id)
    order.status = OrderStatus.CANCELLED
