"""Resoconto giornaliero: fatti → (AI) → messaggio, file salvato, idee nel conto ai_analyst.

Ordine nel workflow serale: prima il paper trading elabora la giornata, poi
questo modulo legge lo stato aggiornato, chiede il commento a Claude, imposta
le idee dell'AI come ordini per la prossima chiusura e invia il messaggio.
"""

from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantlab.analyst.facts import ETF_NAMES, build_facts
from quantlab.analyst.llm import AnalystOutput, ask_claude
from quantlab.paper.store import STATE_DIR, PaperStore

AI_ACCOUNT = "ai_analyst"
BRIEFING_DIR = STATE_DIR / "briefings"
CHANGE_TOLERANCE = 0.02     # sotto questa differenza di peso le idee non generano ordini


@dataclass
class Briefing:
    date: str
    text_html: str
    text_md: str
    facts: dict
    ai: AnalystOutput | None
    ideas_applied: bool


def make_briefing(store: PaperStore, prices: pd.DataFrame, tbill_pct: float | None,
                  use_ai: bool = True, client=None) -> Briefing:
    facts = build_facts(store, prices, tbill_pct)
    ai_portfolio = next((c["posizioni"] for c in facts["conti"] if c["conto"] == AI_ACCOUNT), {})
    ai = None
    if use_ai:
        try:
            ai = ask_claude(facts, ai_portfolio, client=client)
        except Exception as exc:          # l'AI non deve mai bloccare il resoconto
            ai = AnalystOutput(None, problems=[f"analista AI non disponibile: {type(exc).__name__}: {exc}"])
    applied = apply_ideas(store, facts, ai) if ai and ai.ideas is not None else False
    md = render(facts, ai, applied, html_mode=False)
    return Briefing(facts["data"], render(facts, ai, applied, html_mode=True), md, facts, ai, applied)


def apply_ideas(store: PaperStore, facts: dict, ai: AnalystOutput) -> bool:
    """Imposta le idee come ordini del conto ai_analyst (eseguiti alla prossima chiusura)."""
    names = {a["name"] for a in store.accounts()}
    if AI_ACCOUNT not in names:
        return False
    acc = store.account(AI_ACCOUNT)
    if acc["status"] != "attivo":
        return False
    last = store.last_day(AI_ACCOUNT)
    if last is None:
        return False
    current = {a: v / last["equity"] for a, v in last["values"].items()}
    targets = {a: ai.ideas["targets"].get(a, 0.0) for a in acc["spec"]["universe"]}
    if max(abs(targets[a] - current.get(a, 0.0)) for a in targets) < CHANGE_TOLERANCE:
        return False        # portafoglio sostanzialmente invariato: nessun ordine, nessun costo
    store.set_pending(AI_ACCOUNT, last["date"], targets)
    store.log_event(AI_ACCOUNT, last["date"], "idee_ai",
                    "; ".join(f"{e} {w:.0%}: {ai.ideas['motivi'].get(e, '')}"
                              for e, w in ai.ideas["targets"].items()) or "tutto in liquidità")
    return True


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
RISK_EVENTS = ("arresto", "ordini_bloccati", "dati_bloccati", "avviso_dati")


def it(text: str) -> str:
    """Formato numerico italiano nelle righe scritte dal codice: 1,759.5 → 1.759,5."""
    return _NUMBER.sub(lambda m: m.group(0).replace(",", "§").replace(".", ",").replace("§", "."), text)


def render(facts: dict, ai: AnalystOutput | None, applied: bool, html_mode: bool) -> str:
    b = (lambda s: f"<b>{html.escape(it(s))}</b>") if html_mode else (lambda s: f"**{it(s)}**")
    e = (lambda s: html.escape(it(s))) if html_mode else it
    raw = html.escape if html_mode else (lambda s: s)     # testo dell'AI: già in italiano
    day = pd.Timestamp(facts["data"])
    lines = [b(f"QuantLab · {day:%d/%m/%Y}"), ""]

    # 1. Operazioni (scritte dal codice).
    lines.append(b("Operazioni alla prossima chiusura"))
    any_order = False
    for c in facts["conti"]:
        if not c["ordini_prossima_chiusura"]:
            continue
        any_order = True
        lines.append(e(f"{c['strategia']}:"))
        for o in c["ordini_prossima_chiusura"]:
            lines.append(e(f"  • {o['azione'].upper()} {o['etf']} ({ETF_NAMES.get(o['etf'], '')}): "
                           f"peso {o['peso_da']:.0%} → {o['peso_a']:.0%}, circa {o['controvalore']:,.0f}"))
    if not any_order:
        lines.append(e(f"Nessuna. Prossimo ribilanciamento delle strategie: "
                       f"{pd.Timestamp(facts['prossimo_ribilanciamento_strategie']):%d/%m}."))
    lines.append("")

    # 2. Conti simulati (scritto dal codice).
    lines.append(b("Conti simulati"))
    for c in facts["conti"]:
        stato = "" if c["stato"] == "attivo" else " · FERMATO"
        lines.append(e(f"• {c['conto']}: {c['valore']:,.0f} ({c['rendimento_da_avvio']:+.2%} dall'avvio), "
                       f"drawdown {c['drawdown']:.1%}, liquidità {c['liquidita']:.0%}{stato}"))
        for ev in c["eventi_oggi"]:
            icon = "⚠" if ev.startswith(RISK_EVENTS) else "ℹ"
            lines.append(e(f"   {icon} {ev}"))
    lines.append("")

    # 3. Segnali (scritto dal codice).
    sig = facts["segnali"]
    pos = [t for t, s in sig.items() if s["trend"] == "positivo"]
    neg = [t for t, s in sig.items() if s["trend"] == "negativo"]
    lines.append(b("Segnali a 12 mesi"))
    lines.append(e(f"Positivi ({len(pos)}): {', '.join(pos) or '—'} · Negativi ({len(neg)}): {', '.join(neg) or '—'}"))
    near = sorted(sig.items(), key=lambda kv: abs(kv[1]["rend_12m"]))[:2]
    lines.append(e("Più vicini a cambiare segnale: " +
                   ", ".join(f"{t} {s['rend_12m']:+.1%}" for t, s in near)))
    lines.append("")

    # 4. Commento e idee dell'AI.
    if ai is None:
        lines.append(e("Analista AI non attivo (manca la chiave API)."))
    else:
        if ai.commentary:
            lines.append(b("Commento dell'analista AI"))
            lines.append(raw(ai.commentary))
            for w in ai.watch:
                lines.append(raw(f"👀 {w}"))
            lines.append("")
        if ai.ideas is not None:
            lines.append(b("Idee dell'AI (conto simulato ai_analyst, esperimento)"))
            if ai.ideas["targets"]:
                for etf, w in ai.ideas["targets"].items():
                    lines.append(e(f"• {etf} {w:.0%}: ") + raw(ai.ideas["motivi"].get(etf, "")))
            liquidity = 1 - sum(ai.ideas["targets"].values())
            lines.append(e(f"• Liquidità {liquidity:.0%}: ") + raw(ai.ideas.get("motivo_liquidita", "")))
            lines.append(e("Ordini impostati per la prossima chiusura." if applied
                           else "Nessuna modifica rispetto al portafoglio attuale."))
        for p in ai.problems:
            lines.append(e(f"ℹ {p}"))
    lines.append("")
    lines.append(e("Conti simulati, nessun ordine reale. Informazioni generali, non consulenza "
                   "finanziaria personalizzata."))
    return "\n".join(lines)


def save(br: Briefing, folder: Path | None = None) -> Path:
    folder = folder or BRIEFING_DIR
    folder.mkdir(parents=True, exist_ok=True)
    md = folder / f"{br.date}.md"
    md.write_text(br.text_md)
    (folder / f"{br.date}.json").write_text(json.dumps({
        "fatti": br.facts,
        "ai": None if br.ai is None else {"commento": br.ai.commentary, "da_tenere_docchio": br.ai.watch,
                                          "idee": br.ai.ideas, "problemi": br.ai.problems,
                                          "modello": br.ai.model},
        "idee_applicate": br.ideas_applied,
    }, indent=2, ensure_ascii=False))
    return md


def ai_enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
