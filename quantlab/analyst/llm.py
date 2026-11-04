"""Chiamata a Claude per il commento giornaliero e le idee dell'analista AI.

Regole di sicurezza applicate dal codice, non solo chieste nel prompt:
- la risposta deve rispettare uno schema JSON (output strutturato);
- il commento viene scartato se contiene numeri che non compaiono nei fatti;
- le idee vengono validate: solo i 9 ETF, solo posizioni lunghe, al massimo
  5 posizioni da non più del 25%, somma ≤ 100%. Altrimenti vengono ignorate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from quantlab.paper.presets import ETF_9

MODEL = "claude-opus-5-5"
MAX_POSITIONS = 5
MAX_WEIGHT = 0.25

SYSTEM = """Sei l'analista quantitativo di QuantLab, una piattaforma di ricerca personale.
Ogni sera ricevi i fatti del giorno (in JSON) e scrivi in italiano, per un investitore privato.

Il tuo compito ha due parti.

1. COMMENTO (3-6 frasi) e DA TENERE D'OCCHIO (massimo 3 punti brevi).
   Spiega in modo chiaro cosa dicono i segnali delle strategie validate e i conti simulati:
   quali ETF sono in trend positivo o negativo, chi è vicino a cambiare segnale (rendimento a
   12 mesi vicino allo zero), rischi o anomalie segnalati. Usa SOLO numeri presenti nei fatti,
   eventualmente espressi in percentuale con una cifra decimale; non inventare prezzi, livelli,
   notizie o previsioni. Se non c'è nulla di rilevante, dillo in una frase.
   Non dare consigli personalizzati: descrivi cosa fanno le strategie.

2. IDEE per il conto simulato "ai_analyst" (un esperimento separato, misurato nel tempo).
   Proponi il portafoglio che terresti da domani tra questi ETF: SPY, EFA, EEM, IWM, TLT, IEF, GLD,
   DBC, VNQ. Solo posizioni lunghe, al massimo 5 ETF, ciascuno al massimo 25%, il resto in liquidità.
   Basa le idee solo sui fatti forniti (trend, momentum a 1 e 12 mesi, volatilità) e motiva ogni scelta
   in una frase. Se il portafoglio attuale del conto ai_analyst è ancora coerente, riproponilo uguale:
   cambiare spesso costa. Sii onesto sull'incertezza: sono idee da verificare, non certezze."""

SCHEMA = {
    "type": "object",
    "properties": {
        "commento": {"type": "string"},
        "da_tenere_docchio": {"type": "array", "items": {"type": "string"}},
        "idee": {
            "type": "object",
            "properties": {
                "portafoglio": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "etf": {"type": "string", "enum": list(ETF_9)},
                            "peso": {"type": "number"},
                            "motivo": {"type": "string"},
                        },
                        "required": ["etf", "peso", "motivo"],
                        "additionalProperties": False,
                    },
                },
                "motivo_liquidita": {"type": "string"},
                "orizzonte_giorni": {"type": "integer"},
            },
            "required": ["portafoglio", "motivo_liquidita", "orizzonte_giorni"],
            "additionalProperties": False,
        },
    },
    "required": ["commento", "da_tenere_docchio", "idee"],
    "additionalProperties": False,
}


@dataclass
class AnalystOutput:
    commentary: str | None
    watch: list[str] = field(default_factory=list)
    ideas: dict | None = None          # {"targets": {etf: peso}, "motivi": {...}, ...}
    problems: list[str] = field(default_factory=list)
    model: str | None = None


def ask_claude(facts: dict, ai_portfolio: dict, client=None) -> AnalystOutput:
    """Chiede a Claude commento e idee. `client` serve ai test (oggetto con la stessa interfaccia)."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    payload = {"fatti": facts, "portafoglio_attuale_ai_analyst": ai_portfolio}
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                     # in caso di rifiuto, modello alternativo consigliato
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    if response.stop_reason == "refusal":
        return AnalystOutput(None, problems=["richiesta rifiutata dal modello"], model=response.model)
    if response.stop_reason == "max_tokens":
        return AnalystOutput(None, problems=["risposta troncata"], model=response.model)
    text = next((b.text for b in response.content if b.type == "text"), None)
    try:
        data = json.loads(text or "")
    except json.JSONDecodeError:
        return AnalystOutput(None, problems=["risposta non in JSON valido"], model=response.model)
    return check_output(data, facts, model=response.model)


def check_output(data: dict, facts: dict, model: str | None = None) -> AnalystOutput:
    """Applica i controlli su numeri e idee a una risposta già decodificata."""
    out = AnalystOutput(None, model=model)
    allowed = allowed_numbers(facts)
    text_parts = [data.get("commento", "")] + list(data.get("da_tenere_docchio", []))[:3]
    unknown = sorted({n for t in text_parts for n in numbers_in(t) if not _matches(n, allowed)})
    if unknown:
        out.problems.append(f"commento scartato: numeri non presenti nei fatti {unknown[:5]}")
    else:
        out.commentary = data.get("commento", "").strip()
        out.watch = [w.strip() for w in data.get("da_tenere_docchio", [])[:3] if w.strip()]
    out.ideas, problems = validate_ideas(data.get("idee", {}))
    out.problems += problems
    return out


def validate_ideas(ideas: dict) -> tuple[dict | None, list[str]]:
    rows = ideas.get("portafoglio", [])
    problems = []
    targets = {}
    for r in rows:
        etf, w = r.get("etf"), r.get("peso")
        if etf not in ETF_9 or not isinstance(w, (int, float)) or w <= 0:
            problems.append(f"idea non valida ignorata: {r}")
            continue
        targets[etf] = targets.get(etf, 0.0) + float(w)
    if len(targets) > MAX_POSITIONS:
        return None, problems + [f"idee scartate: {len(targets)} posizioni, massimo {MAX_POSITIONS}"]
    if any(w > MAX_WEIGHT + 1e-9 for w in targets.values()):
        return None, problems + [f"idee scartate: un peso supera il {MAX_WEIGHT:.0%}"]
    if sum(targets.values()) > 1 + 1e-9:
        return None, problems + ["idee scartate: somma dei pesi oltre il 100%"]
    return {
        "targets": {e: round(w, 4) for e, w in targets.items()},
        "motivi": {r["etf"]: r.get("motivo", "") for r in rows if r.get("etf") in targets},
        "motivo_liquidita": ideas.get("motivo_liquidita", ""),
        "orizzonte_giorni": ideas.get("orizzonte_giorni"),
    }, problems


# --- Controllo dei numeri -----------------------------------------------------------------

_NUM = re.compile(r"(?<![A-Za-z])[-+−]?\d+(?:[.,]\d+)?")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text or ""):
        s = m.replace("−", "-").replace(",", ".")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def allowed_numbers(facts: dict) -> set[float]:
    """Numeri ammessi: valori dei fatti, anche in percentuale, più date e piccoli conteggi."""
    vals: set[float] = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, bool):
            return
        elif isinstance(x, (int, float)):
            vals.update({float(x), round(x * 100, 1), round(x * 100, 0), round(x, 2),
                         abs(float(x)), abs(round(x * 100, 1)), abs(round(x * 100, 0))})
        elif isinstance(x, str):
            vals.update(numbers_in(x))
    walk(facts)
    vals.update(float(i) for i in range(0, 32))      # giorni, mesi, conteggi ("9 ETF", "12 mesi")
    vals.update(float(y) for y in range(2000, 2101))  # anni
    return vals


def _matches(n: float, allowed: set[float]) -> bool:
    return any(abs(n - a) <= 0.051 or abs(abs(n) - abs(a)) <= 0.051 for a in allowed)
