"""Comandi del paper trading.

    python -m quantlab.paper create            # crea il conto "rp_momentum" (risk parity + momentum)
    python -m quantlab.paper run               # elabora i giorni nuovi per tutti i conti attivi
    python -m quantlab.paper status            # valore, posizioni, ultimi eventi
    python -m quantlab.paper halt rp_momentum --liquidate   # pulsante di emergenza
    python -m quantlab.paper resume rp_momentum
    python -m quantlab.paper ensure            # crea i conti predefiniti mancanti (usato da GitHub Actions)
    python -m quantlab.paper export            # scrive i CSV leggibili in state/paper/

`run` va eseguito una volta al giorno dopo le 16:30 di New York (22:30 in Italia).
Con l'automazione attiva (state/automation.json) lo fa GitHub Actions ogni sera.
"""

from __future__ import annotations

import argparse

import pandas as pd

from quantlab.paper import PaperStore, RiskLimits, create_account, halt, resume, run_account
from quantlab.paper.presets import PRESETS


def export_csv(store: PaperStore, out=None):
    """CSV leggibili su GitHub: valori giornalieri, ordini ed eventi di ogni conto."""
    from pathlib import Path

    from quantlab.paper.store import STATE_DIR
    out = Path(out) if out else STATE_DIR / "paper"
    out.mkdir(parents=True, exist_ok=True)
    for acc in store.accounts():
        n = acc["name"]
        h = store.history(n)
        if len(h):
            h.round(4).to_csv(out / f"{n}_daily.csv", index_label="date")
        o = store.orders(n)
        if len(o):
            o.drop(columns=["account"]).round(4).to_csv(out / f"{n}_orders.csv", index=False)
        ev = store.events(n)
        if len(ev):
            ev.drop(columns=["account"]).to_csv(out / f"{n}_events.csv", index=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Paper trading di QuantLab.")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--name", default="rp_momentum")
    c.add_argument("--preset", default="rp_momentum", choices=sorted(PRESETS))
    c.add_argument("--capital", type=float, default=10_000.0)
    sub.add_parser("run")
    sub.add_parser("status")
    h = sub.add_parser("halt")
    h.add_argument("name")
    h.add_argument("--liquidate", action="store_true", help="vende tutto alla prossima chiusura")
    r = sub.add_parser("resume")
    r.add_argument("name")
    sub.add_parser("ensure")
    e = sub.add_parser("export")
    e.add_argument("--out", default=None, help="cartella di destinazione (default: state/paper)")
    args = p.parse_args()

    # Con l'automazione attiva lo stato condiviso (state/paper.sqlite) lo scrive solo GitHub Actions:
    # scriverlo anche in locale creerebbe due storie diverse e conflitti al git pull.
    import os

    from quantlab.paper.store import automation_info, default_paper_db
    writes = args.cmd in {"run", "create", "halt", "resume", "ensure", "export"}
    on_github = os.environ.get("GITHUB_ACTIONS") == "true"
    if writes and automation_info() and not on_github and not os.environ.get("QUANTLAB_PAPER_DB"):
        raise SystemExit(
            "Automazione attiva: lo stato del paper trading è scritto solo da GitHub Actions.\n"
            "Per elaborare, fermare o riattivare un conto: GitHub → Actions → Paper trading giornaliero "
            "→ Run workflow.\nPer provare in locale senza toccare lo stato condiviso: "
            "QUANTLAB_PAPER_DB=data/paper.sqlite python -m quantlab.paper ..."
        )
    store = PaperStore()

    if args.cmd == "ensure":
        existing = {a["name"] for a in store.accounts()}
        for key, preset in PRESETS.items():
            if key not in existing:
                create_account(store, key, preset["spec"], 10_000.0, RiskLimits(), preset["note"])
                print(f"Conto {key!r} creato.")
        print(f"Conti presenti: {', '.join(sorted({a['name'] for a in store.accounts()}))}")
    elif args.cmd == "export":
        out = export_csv(store, args.out)
        print(f"CSV scritti in {out}")
    elif args.cmd == "create":
        preset = PRESETS[args.preset]
        create_account(store, args.name, preset["spec"], args.capital, RiskLimits(), preset["note"])
        print(f"Conto {args.name!r} creato con {args.capital:,.0f} di capitale simulato.")
        print("Ora esegui: python -m quantlab.paper run")
    elif args.cmd == "run":
        for acc in store.accounts():
            s = run_account(store, acc["name"])
            days = ", ".join(str(d.date()) for d in s.processed) or "nessuno"
            print(f"[{acc['name']}] stato {s.status} · giorni elaborati: {days} · "
                  f"ordini eseguiti {s.filled}, bloccati {s.blocked}"
                  + (f" · valore {s.equity:,.2f}" if s.equity is not None else ""))
            for m in s.messages:
                print(f"   - {m}")
    elif args.cmd == "status":
        for acc in store.accounts():
            last = store.last_day(acc["name"])
            print(f"\n[{acc['name']}] {acc['status']} · capitale iniziale {acc['initial_capital']:,.0f}")
            if not last:
                print("   nessun giorno elaborato: esegui `python -m quantlab.paper run`")
                continue
            ret = last["equity"] / acc["initial_capital"] - 1
            print(f"   {last['date'].date()}: valore {last['equity']:,.2f} ({ret:+.2%}), "
                  f"liquidità {last['cash'] / last['equity']:.0%}")
            pos = {a: v / last["equity"] for a, v in last["values"].items() if v > 0}
            if pos:
                print("   posizioni: " + ", ".join(f"{a} {w:.0%}" for a, w in
                                                  sorted(pos.items(), key=lambda x: -x[1])))
            pend = store.pending(acc["name"])
            if pend:
                print(f"   ordini in attesa (segnale del {pend['signal_date'].date()}): esecuzione alla prossima chiusura")
            ev = store.events(acc["name"])
            for _, e in ev.tail(5).iterrows():
                print(f"   {e['date']} {e['kind']}: {e['message']}")
    elif args.cmd == "halt":
        halt(store, args.name, args.liquidate)
        print(f"Conto {args.name!r} fermato" + (" (liquidazione alla prossima chiusura)." if args.liquidate else "."))
    elif args.cmd == "resume":
        resume(store, args.name)
        print(f"Conto {args.name!r} riattivato.")


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    main()
