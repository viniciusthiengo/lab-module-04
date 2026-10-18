"""Stock reservations, so that two orders cannot sell the same copy."""

import threading

LOW_STOCK_THRESHOLD = 5


class OutOfStock(Exception):
    """A book of the order has fewer copies than requested."""


class Inventory:
    def __init__(self, stock: dict[int, int]):
        self._stock = dict(stock)
        self._reserved: dict[str, dict[int, int]] = {}
        self._lock = threading.Lock()

    def reserve(self, order_id: str, items: dict[int, int]) -> None:
        """Reserves every item of an order, or none: raises OutOfStock when a book lacks
        copies."""
        with self._lock:
            for book_id, quantity in items.items():
                if self._stock.get(book_id, 0) < quantity:
                    raise OutOfStock(book_id)
            for book_id, quantity in items.items():
                self._stock[book_id] -= quantity
            self._reserved[order_id] = dict(items)

    def release(self, order_id: str) -> None:
        """Gives back the copies reserved for an order (payment failed or order cancelled)."""
        with self._lock:
            for book_id, quantity in self._reserved.pop(order_id, {}).items():
                self._stock[book_id] += quantity

    def low_stock(self) -> list[int]:
        """Books with fewer than LOW_STOCK_THRESHOLD copies left."""
        with self._lock:
            return [book for book, count in self._stock.items() if count < LOW_STOCK_THRESHOLD]
