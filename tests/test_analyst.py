"""Test dell'analista AI con un finto client: nessuna chiamata reale, nessun costo."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantlab.analyst.briefing import AI_ACCOUNT, make_briefing, save
from quantlab.analyst.llm import check_output, numbers_in, validate_ideas
from quantlab.paper import PaperStore, RiskLimits, create_account, run_account
from quantlab.paper.presets import ETF_9, PRESETS

IDX = pd.bdate_range("2023-01-02", periods=420)


def prices(seed=0):
    rng = np.random.default_rng(seed)
    drift = np.linspace(-0.0004, 0.0008, len(ETF_9))
    r = rng.normal(drift, 0.01, (len(IDX), len(ETF_9)))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=IDX, columns=ETF_9)


class FakeClient:
    """Imita client.beta.messages.create restituendo una risposta preparata."""

    def __init__(self, payload=None, stop_reason="end_turn", raises=None):
        self.payload, self.stop_reason, self.raises = payload, stop_reason, raises
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        text = json.dumps(self.payload) if self.payload is not None else ""
        return SimpleNamespace(stop_reason=self.stop_reason, model="claude-opus-5-5",
                               content=[SimpleNamespace(type="text", text=text)])


def answer(commento="Sei ETF su nove sono in trend positivo.", ideas=None):
    return {"commento": commento, "da_tenere_docchio": ["Oro vicino a cambiare segnale."],
            "idee": ideas or {"portafoglio": [{"etf": "SPY", "peso": 0.2, "motivo": "trend positivo"},
                                              {"etf": "DBC", "peso": 0.15, "motivo": "momentum forte"}],
                              "motivo_liquidita": "prudenza", "orizzonte_giorni": 20}}


@pytest.fixture
def store(tmp_path):
    s = PaperStore(tmp_path / "paper.sqlite")
    p = prices()
    for key in ("rp_momentum", AI_ACCOUNT):
        spec = PRESETS[key]["spec"] | {"cash": "zero"}
        create_account(s, key, spec, 10_000, RiskLimits())
        run_account(s, key, prices=p.iloc[:400], today=p.index[399])
    return s


# --- Controlli sulla risposta -------------------------------------------------------------

def test_numbers_are_parsed_in_italian_format():
    assert numbers_in("drawdown -2,4% e 12 mesi, 1.5") == [-2.4, 12.0, 1.5]


def test_invented_numbers_discard_commentary():
    facts = {"segnali": {"SPY": {"rend_12m": 0.111}}}
    ok = check_output(answer("SPY ha fatto +11,1% in 12 mesi."), facts)
    assert ok.commentary and not ok.problems
    bad = check_output(answer("SPY arriverà a 812 entro dicembre."), facts)
    assert bad.commentary is None and "numeri non presenti" in bad.problems[0]
    assert bad.ideas is not None                     # le idee restano valide e separate


@pytest.mark.parametrize("portfolio,reason", [
    ([{"etf": "TSLA", "peso": 0.1, "motivo": ""}], None),               # ETF fuori universo: ignorato
    ([{"etf": "SPY", "peso": 0.4, "motivo": ""}], "supera"),
    ([{"etf": e, "peso": 0.1, "motivo": ""} for e in ETF_9[:6]], "posizioni"),
    ([{"etf": e, "peso": 0.25, "motivo": ""} for e in ETF_9[:5]], "oltre il 100%"),
])
def test_invalid_ideas_are_rejected(portfolio, reason):
    ideas, problems = validate_ideas({"portafoglio": portfolio, "motivo_liquidita": "", "orizzonte_giorni": 5})
    if reason is None:
        assert ideas["targets"] == {} and problems
    else:
        assert ideas is None and reason in problems[-1]


# --- Resoconto completo -------------------------------------------------------------------

def test_briefing_with_ai_sets_ideas_as_orders(store, tmp_path):
    client = FakeClient(answer())
    br = make_briefing(store, prices().iloc[:400], 4.0, client=client)
    assert br.ideas_applied
    pend = store.pending(AI_ACCOUNT)
    assert pend["targets"]["SPY"] == 0.2 and pend["targets"]["GLD"] == 0.0
    req = client.calls[0]
    assert req["model"] == "claude-opus-5-5" and req["output_config"]["format"]["type"] == "json_schema"
    assert "Commento dell'analista AI" in br.text_md and "<b>" in br.text_html
    path = save(br, tmp_path / "b")
    assert path.exists() and json.loads(path.with_suffix(".json").read_text())["idee_applicate"]
    # Il giorno dopo il paper trading esegue le idee alla chiusura
    p = prices()
    run_account(store, AI_ACCOUNT, prices=p.iloc[:401], today=p.index[400])
    last = store.last_day(AI_ACCOUNT)
    assert last["values"]["SPY"] / last["equity"] == pytest.approx(0.2, abs=0.01)


def test_unchanged_portfolio_creates_no_orders(store):
    make_briefing(store, prices().iloc[:400], 4.0, client=FakeClient(answer()))
    p = prices()
    run_account(store, AI_ACCOUNT, prices=p.iloc[:401], today=p.index[400])
    br = make_briefing(store, p.iloc[:401], 4.0, client=FakeClient(answer()))
    assert not br.ideas_applied and store.pending(AI_ACCOUNT) is None


@pytest.mark.parametrize("client,expected", [
    (FakeClient(stop_reason="refusal"), "rifiutata"),
    (FakeClient(raises=RuntimeError("rete assente")), "non disponibile"),
    (FakeClient(payload=None), "JSON"),
])
def test_ai_failures_never_block_the_briefing(store, client, expected):
    br = make_briefing(store, prices().iloc[:400], 4.0, client=client)
    assert "Operazioni alla prossima chiusura" in br.text_md and expected in br.text_md
    assert store.pending(AI_ACCOUNT) is None


def test_briefing_without_ai(store):
    br = make_briefing(store, prices().iloc[:400], None, use_ai=False)
    assert br.ai is None and "manca la chiave API" in br.text_md
    assert "Segnali a 12 mesi" in br.text_md
