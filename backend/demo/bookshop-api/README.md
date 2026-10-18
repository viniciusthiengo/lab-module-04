# Bookshop API

The HTTP API of a small online bookstore: catalog search, carts with coupons, orders paid by card, and e-mail notifications.

## Running

    pip install -r requirements.txt
    uvicorn bookshop.api:app --reload

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///bookshop.db` | Database connection string. |
| `JWT_SECRET` | none (required) | Secret that signs the access tokens. |
| `TOKEN_TTL_MINUTES` | `30` | How long an access token is valid. |
| `FREE_SHIPPING_THRESHOLD` | `50.00` | Order subtotal from which shipping is free. |
| `SMTP_HOST` | `localhost` | Mail server of the notifications. |

## Payments

Card payments go through the gateway in `bookshop/payments.py`. Timeouts are retried; declined cards are not.
