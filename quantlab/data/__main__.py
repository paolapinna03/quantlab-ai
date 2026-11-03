"""Riga di comando per scaricare e ispezionare i dati.

Esempi:
    python -m quantlab.data SPY
    python -m quantlab.data SPY QQQ TLT GLD --refresh
"""

from __future__ import annotations

import argparse

from quantlab.data import cache_info, load_prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Scarica e controlla i prezzi giornalieri.")
    parser.add_argument("tickers", nargs="+", help="es. SPY QQQ TLT")
    parser.add_argument("--refresh", action="store_true", help="riscarica ignorando la cache")
    args = parser.parse_args()

    for ticker in args.tickers:
        df = load_prices(ticker, refresh=args.refresh)
        info = cache_info(ticker) or {}
        print(f"\n=== {ticker.upper()} ===")
        print(f"periodo:     {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} giorni)")
        print(f"scaricato:   {info.get('fetched_at', '?')}  (fonte: {info.get('source', '?')})")
        print(f"ultima chiusura: {df['close'].iloc[-1]:.2f}")
        warnings = info.get("warnings", [])
        print(f"avvisi:      {len(warnings)}")
        for w in warnings[:5]:
            print(f"  - {w}")
        if len(warnings) > 5:
            print(f"  ... e altri {len(warnings) - 5}")


if __name__ == "__main__":
    main()
