"""Shopping carts: items, coupons, shipping and totals."""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from bookshop.catalog import Book
from bookshop.config import Settings

MAX_QUANTITY_PER_BOOK = 10


class CartError(Exception):
    """An invalid change to a cart."""


@dataclass(frozen=True)
class Coupon:
    code: str
    percent_off: int = 0
    amount_off: Decimal = Decimal("0")
    min_subtotal: Decimal = Decimal("0")
    expires_on: date | None = None


@dataclass
class Cart:
    settings: Settings
    items: dict[int, tuple[Book, int]] = field(default_factory=dict)
    coupon: Coupon | None = None

    def add_item(self, book: Book, quantity: int = 1) -> None:
        """Adds ``quantity`` copies: at most MAX_QUANTITY_PER_BOOK per book, and never more
        than the stock."""
        if quantity < 1:
            raise CartError("quantity must be at least 1")
        current = self.items.get(book.id, (book, 0))[1]
        new_quantity = current + quantity
        if new_quantity > MAX_QUANTITY_PER_BOOK:
            raise CartError(f"at most {MAX_QUANTITY_PER_BOOK} copies per book")
        if new_quantity > book.stock:
            raise CartError(f"only {book.stock} copies in stock")
        self.items[book.id] = (book, new_quantity)

    def remove_item(self, book_id: int) -> None:
        self.items.pop(book_id, None)

    def subtotal(self) -> Decimal:
        return sum(
            (book.price * quantity for book, quantity in self.items.values()), Decimal("0")
        )

    def apply_coupon(self, coupon: Coupon, today: date) -> None:
        """Applies ``coupon`` if it has not expired and the subtotal reaches its minimum.
        A cart has one coupon: a new one replaces the previous."""
        if coupon.expires_on is not None and coupon.expires_on < today:
            raise CartError("coupon expired")
        if self.subtotal() < coupon.min_subtotal:
            raise CartError(f"coupon requires a subtotal of {coupon.min_subtotal}")
        self.coupon = coupon

    def discount(self) -> Decimal:
        """The coupon's percentage plus its fixed amount, never more than the subtotal."""
        if self.coupon is None:
            return Decimal("0")
        subtotal = self.subtotal()
        percent = subtotal * self.coupon.percent_off / 100
        return min(subtotal, percent + self.coupon.amount_off)

    def shipping(self) -> Decimal:
        """Free from the free-shipping threshold (after discounts), a flat fee below it."""
        if not self.items:
            return Decimal("0")
        if self.subtotal() - self.discount() >= self.settings.free_shipping_threshold:
            return Decimal("0")
        return self.settings.shipping_fee

    def total(self) -> Decimal:
        return (self.subtotal() - self.discount() + self.shipping()).quantize(Decimal("0.01"))
