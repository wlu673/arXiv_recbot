import base64
import hashlib
import hmac
import json
import time
import urllib.request


def _sign_webhook(timestamp: str, secret: str) -> str:
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256)
    return base64.b64encode(hmac_code.digest()).decode("utf-8")


def send_feishu_text(webhook_url: str, text: str, secret: str | None = None) -> None:
    payload = {
        "msg_type": "text",
        "content": {"text": text},
    }
    if secret:
        timestamp = str(int(time.time()))
        payload["timestamp"] = timestamp
        payload["sign"] = _sign_webhook(timestamp, secret)

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()
