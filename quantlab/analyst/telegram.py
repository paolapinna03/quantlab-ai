"""Invio dei messaggi su Telegram (Bot API, solo libreria standard).

Servono due valori, salvati come secret su GitHub e mai nel codice:
- TELEGRAM_BOT_TOKEN: il token del bot creato con @BotFather;
- TELEGRAM_CHAT_ID: l'identificativo della tua chat con il bot.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

API = "https://api.telegram.org/bot{token}/{method}"
MAX_LEN = 4000          # Telegram accetta al massimo 4096 caratteri per messaggio


def configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


def send(text: str, token: str | None = None, chat_id: str | None = None) -> None:
    """Invia `text` (HTML semplice di Telegram), spezzandolo se è troppo lungo."""
    token = token or os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = chat_id or os.environ["TELEGRAM_CHAT_ID"]
    for chunk in _chunks(text):
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": chunk, "parse_mode": "HTML",
                                       "disable_web_page_preview": "true"}).encode()
        req = urllib.request.Request(API.format(token=token, method="sendMessage"), data=data)
        with urllib.request.urlopen(req, timeout=20) as r:
            body = json.loads(r.read())
        if not body.get("ok"):
            raise RuntimeError(f"Telegram ha rifiutato il messaggio: {body.get('description')}")


def find_chat_id(token: str) -> list[tuple[str, str]]:
    """Chat che hanno scritto al bot di recente: [(chat_id, nome)]. Serve una volta, per la configurazione."""
    with urllib.request.urlopen(API.format(token=token, method="getUpdates"), timeout=20) as r:
        body = json.loads(r.read())
    chats = {}
    for u in body.get("result", []):
        chat = (u.get("message") or u.get("edited_message") or {}).get("chat")
        if chat:
            chats[str(chat["id"])] = chat.get("first_name") or chat.get("title") or ""
    return list(chats.items())


def _chunks(text: str) -> list[str]:
    if len(text) <= MAX_LEN:
        return [text]
    parts, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > MAX_LEN:
            parts.append(cur)
            cur = ""
        cur += line + "\n"
    return parts + ([cur] if cur else [])
