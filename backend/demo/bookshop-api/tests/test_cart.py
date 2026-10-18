from datetime import date
from decimal import Decimal

import pytest

from bookshop.cart import Cart, CartError, Coupon
from bookshop.catalog import Book
from bookshop.config import Settings

SETTINGS = Settings(database_url="sqlite://", jwt_secret="test")
BOOK = Book(id=1, title="Dune", author="Frank Herbert", price=Decimal("30.00"), stock=3)


def test_free_shipping_from_the_threshold():
    cart = Cart(SETTINGS)
    cart.add_item(BOOK, 2)
    assert cart.shipping() == Decimal("0")


def test_expired_coupons_are_refused():
    cart = Cart(SETTINGS)
    cart.add_item(BOOK)
    expired = Coupon("OLD", percent_off=10, expires_on=date(2020, 1, 1))
    with pytest.raises(CartError, match="expired"):
        cart.apply_coupon(expired, date(2026, 1, 1))


def test_quantity_is_limited_by_stock():
    cart = Cart(SETTINGS)
    with pytest.raises(CartError, match="in stock"):
        cart.add_item(BOOK, 4)
