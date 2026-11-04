"""Analista AI giornaliero.

    python -m quantlab.analyst run          # resoconto + idee AI + Telegram (usato da GitHub Actions)
    python -m quantlab.analyst preview      # mostra il resoconto senza AI, senza inviare né salvare
    python -m quantlab.analyst chat-id      # trova il tuo TELEGRAM_CHAT_ID (serve TELEGRAM_BOT_TOKEN)
    python -m quantlab.analyst test-telegram

Senza ANTHROPIC_API_KEY il resoconto viene prodotto lo stesso, senza commento e idee.
Senza TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID viene solo salvato in state/briefings/.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from quantlab.analyst import telegram
from quantlab.analyst.briefing import ai_enabled, make_briefing, save
from quantlab.data import load_prices
from quantlab.data.rates import load_tbill_yield
from quantlab.paper.presets import ETF_9
from quantlab.paper.store import PaperStore


def load_inputs():
    prices = pd.DataFrame({t: load_prices(t)["close"] for t in ETF_9})
    try:
        y = load_tbill_yield()
        tbill = float(y.iloc[-1])
    except Exception:
        tbill = None
    return prices, tbill


def main() -> None:
    p = argparse.ArgumentParser(description="Analista AI di QuantLab.")
    p.add_argument("cmd", choices=["run", "preview", "chat-id", "test-telegram"])
    args = p.parse_args()

    if args.cmd == "chat-id":
        token = os.environ.get("TELEGRAM_BOT_TOKEN") or input("Token del bot: ").strip()
        chats = telegram.find_chat_id(token)
        if not chats:
            print("Nessuna chat trovata: scrivi un messaggio qualsiasi al tuo bot su Telegram e riprova.")
        for cid, name in chats:
            print(f"TELEGRAM_CHAT_ID = {cid}   ({name})")
        return
    if args.cmd == "test-telegram":
        telegram.send("<b>QuantLab</b>: collegamento con Telegram riuscito ✅")
        print("Messaggio di prova inviato.")
        return

    store = PaperStore()
    prices, tbill = load_inputs()
    if args.cmd == "preview":
        print(make_briefing(store, prices, tbill, use_ai=False).text_md)
        return

    br = make_briefing(store, prices, tbill, use_ai=ai_enabled())
    path = save(br)
    print(br.text_md)
    print(f"\nResoconto salvato in {path}")
    if telegram.configured():
        try:
            telegram.send(br.text_html)
            print("Inviato su Telegram.")
        except Exception as exc:
            print(f"Invio Telegram non riuscito: {exc}", file=sys.stderr)
    else:
        print("Telegram non configurato: messaggio non inviato.")


if __name__ == "__main__":
    main()
