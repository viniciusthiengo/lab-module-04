"""Password hashing and signed access tokens."""

import base64
import hashlib
import hmac
import json
import os
import time

PBKDF2_ITERATIONS = 210_000
SALT_BYTES = 16


class InvalidToken(Exception):
    """The token is malformed, tampered with, or expired."""


def hash_password(password: str) -> str:
    """Returns ``salt$hash``: PBKDF2-HMAC-SHA256 with a random 16-byte salt."""
    salt = os.urandom(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Recomputes the hash with the stored salt and compares in constant time."""
    salt_hex, digest_hex = stored.split("$", 1)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(digest.hex(), digest_hex)


def create_token(user_id: int, secret: str, ttl_minutes: int) -> str:
    """A compact signed token: base64(payload).HMAC-SHA256 signature, valid ttl_minutes."""
    payload = {"sub": user_id, "exp": int(time.time()) + ttl_minutes * 60}
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    signature = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{signature}"


def decode_token(token: str, secret: str) -> int:
    """Returns the user id of a valid token; raises ``InvalidToken`` otherwise."""
    try:
        body, signature = token.split(".", 1)
    except ValueError:
        raise InvalidToken("malformed token") from None
    expected = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise InvalidToken("bad signature")
    payload = json.loads(base64.urlsafe_b64decode(body))
    if payload["exp"] < time.time():
        raise InvalidToken("token expired")
    return payload["sub"]
