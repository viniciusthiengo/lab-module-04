"""Settings of the bookshop, read from environment variables."""

import os
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Settings:
    database_url: str
    jwt_secret: str
    token_ttl_minutes: int = 30
    free_shipping_threshold: Decimal = Decimal("50.00")
    shipping_fee: Decimal = Decimal("4.99")
    smtp_host: str = "localhost"

    @classmethod
    def from_env(cls) -> "Settings":
        secret = os.environ.get("JWT_SECRET")
        if not secret:
            raise RuntimeError("JWT_SECRET must be set")
        return cls(
            database_url=os.environ.get("DATABASE_URL", "sqlite:///bookshop.db"),
            jwt_secret=secret,
            token_ttl_minutes=int(os.environ.get("TOKEN_TTL_MINUTES", "30")),
            free_shipping_threshold=Decimal(os.environ.get("FREE_SHIPPING_THRESHOLD", "50.00")),
            smtp_host=os.environ.get("SMTP_HOST", "localhost"),
        )
