"""E-mail notifications, limited per recipient."""

import smtplib
import time
from collections import defaultdict, deque
from email.message import EmailMessage

MAX_EMAILS_PER_HOUR = 5

CONFIRMATION_TEMPLATE = """Hello,

Your order {order_id} is confirmed. Total: {total}.

Thank you for shopping with us!
"""


class Notifier:
    def __init__(self, smtp_host: str, sender: str = "orders@bookshop.example", clock=time.time):
        self._smtp_host = smtp_host
        self._sender = sender
        self._clock = clock
        self._sent: dict[str, deque[float]] = defaultdict(deque)

    def order_confirmation(self, order) -> bool:
        body = CONFIRMATION_TEMPLATE.format(order_id=order.id, total=order.total)
        return self.send(order.user_email, "Your order is confirmed", body)

    def send(self, to: str, subject: str, body: str) -> bool:
        """Sends an e-mail unless the recipient already got MAX_EMAILS_PER_HOUR in the last
        hour; returns whether it was sent."""
        now = self._clock()
        recent = self._sent[to]
        while recent and now - recent[0] > 3600:
            recent.popleft()
        if len(recent) >= MAX_EMAILS_PER_HOUR:
            return False
        message = EmailMessage()
        message["From"], message["To"], message["Subject"] = self._sender, to, subject
        message.set_content(body)
        with smtplib.SMTP(self._smtp_host) as smtp:
            smtp.send_message(message)
        recent.append(now)
        return True
