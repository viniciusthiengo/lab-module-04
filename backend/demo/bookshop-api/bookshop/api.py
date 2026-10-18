"""HTTP API of the bookshop."""

from decimal import Decimal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from bookshop.auth import InvalidToken, decode_token
from bookshop.catalog import paginate, search_books
from bookshop.config import Settings
from bookshop.rate_limit import RateLimiter

settings = Settings.from_env()
app = FastAPI(title="Bookshop API")
limiter = RateLimiter(capacity=20, rate=5)
BOOKS = []  # loaded from the database at startup


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Answers 429 with a Retry-After header when the client's bucket is empty."""
    allowed, retry_after = limiter.check(request.client.host)
    if not allowed:
        return JSONResponse(
            {"error": "Too many requests"},
            status_code=429,
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
    return await call_next(request)


def current_user(authorization: str = Header(default="")) -> int:
    """The user id of the bearer token; 401 when it is missing, invalid or expired."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    try:
        return decode_token(authorization.removeprefix("Bearer "), settings.jwt_secret)
    except InvalidToken as error:
        raise HTTPException(status_code=401, detail=str(error)) from error


@app.get("/books")
def list_books(
    q: str = "",
    author: str | None = None,
    max_price: Decimal | None = None,
    page: int = 1,
    per_page: int = 20,
):
    return paginate(search_books(BOOKS, q, author, max_price), page, per_page)


@app.get("/me/orders")
def my_orders(user_id: int = Depends(current_user)):
    return {"user_id": user_id, "orders": []}
