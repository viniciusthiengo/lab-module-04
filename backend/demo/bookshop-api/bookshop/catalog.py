"""The book catalog: search and pagination."""

from dataclasses import dataclass
from decimal import Decimal

MAX_PAGE_SIZE = 50


@dataclass(frozen=True)
class Book:
    id: int
    title: str
    author: str
    price: Decimal
    stock: int


def search_books(books, query="", author=None, max_price=None):
    """Books whose title contains ``query`` (case-insensitive), optionally filtered by
    author and maximum price, sorted by title."""
    needle = query.casefold()
    found = [
        book
        for book in books
        if needle in book.title.casefold()
        and (author is None or book.author == author)
        and (max_price is None or book.price <= max_price)
    ]
    return sorted(found, key=lambda book: book.title)


def paginate(items, page=1, per_page=20):
    """One page of ``items``; ``per_page`` is capped at MAX_PAGE_SIZE."""
    per_page = max(1, min(per_page, MAX_PAGE_SIZE))
    start = (max(page, 1) - 1) * per_page
    return {
        "items": items[start : start + per_page],
        "page": page,
        "per_page": per_page,
        "total": len(items),
    }
