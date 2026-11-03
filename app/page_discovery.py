"""Pagina Strategy Discovery: risultati delle ricerche automatiche di strategie.

La ricerca si avvia da terminale (python -m quantlab.discovery), perché è
lunga e aggiunge tentativi al registro: deve essere una scelta consapevole,
non un clic. Questa pagina mostra i risultati salvati.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from charts import PLOTLY_CONFIG, candidates_chart, color_map, dumbbell_chart, equity_chart
from styles import card_title, criteria_table, esc, header, verdict_block
from quantlab.config import HOLDOUT_START
from quantlab.discovery.report import load_session
from quantlab.experiments import ExperimentRegistry, ExperimentSpec, run_experiment
from quantlab.metrics import equity_curve
from quantlab.validation import Check

registry = ExperimentRegistry()
names = list(dict.fromkeys(s["name"] for s in registry.discoveries()))

if not names:
    st.html(header("Strategy Discovery", "Ricerca automatica di strategie",
                   "Nessuna sessione ancora eseguita.", [("Holdout bloccato", "lock")]))
    st.info("Avvia la prima ricerca da terminale:\n\n`python -m quantlab.discovery`\n\n"
            "Prova 60 combinazioni di regole su 16 ETF e conferma le migliori su dati mai usati.")
    st.stop()

with st.sidebar:
    name = st.selectbox("Sessione", names, help="Ogni sessione è una famiglia di tentativi nel registro.")

data = load_session(registry, name)
cfg, search, confirmed = data["config"], data["search"], data["confirmed"]
promoted = int((confirmed["verdict"] == "PROMOSSA").sum()) if not confirmed.empty else 0


# --- Intestazione e imbuto ----------------------------------------------------------

st.html(header(
    eyebrow=f"Strategy Discovery · sessione {name}",
    title="Ricerca automatica di strategie swing",
    meta=(f"<b>{len(cfg['universe'])} ETF</b> · training <b>fino al {cfg['train_end']}</b> · "
          f"conferma <b>{cfg['confirm_start']} → {cfg['confirm_end']}</b> · seme {cfg['seed']} · "
          f"costi {cfg['cost_bps']:g} bp + slippage"),
    badges=[(f"{'✓' if promoted else '✗'} {promoted} promosse", "pass" if promoted else "fail"),
            (f"Holdout {HOLDOUT_START[:4]}+ bloccato", "lock")],
))

k = st.columns(4)
k[0].metric("Spazio di ricerca", f"{cfg['space_size']:,}".replace(",", "."), "combinazioni possibili",
            delta_color="off", border=True)
k[1].metric("Candidate testate", f"{len(search)}", "solo sul training", delta_color="off", border=True)
k[2].metric("Confermate", f"{len(confirmed)}", f"le migliori {cfg['top_k']} con ≥ {cfg['min_trades']} operazioni",
            delta_color="off", border=True)
k[3].metric("Promosse", f"{promoted}", "superano tutti i criteri", delta_color="off", border=True)

if confirmed.empty:
    st.warning(f"Nessuna candidata ha raggiunto la conferma (servono almeno {cfg['min_trades']} "
               "operazioni nel training). Controlla la tabella completa qui sotto.")
    st.dataframe(search, width="stretch", hide_index=True)
    st.stop()


# --- Classifica di conferma + training vs conferma ----------------------------------------

left, right = st.columns([1.6, 1], gap="medium")
with left, st.container(border=True):
    st.html(card_title("Classifica di conferma", "dati 2017–2022, mai usati per scegliere"))
    rows = "".join(
        f"""<tr>
          <td><span class="name">{r['rank']} {esc(r['description'])}</span></td>
          <td>{r['train_sharpe']:.2f}</td>
          <td>{r['confirm_sharpe']:.2f}</td>
          <td>{r['confirm_max_drawdown']:.1%}</td>
          <td>{int(r['confirm_trades'] or 0)}</td>
          <td>{r['checks_passed']}/{len(r['checks'])}</td>
          <td><span class="ql-status {'ok' if r['verdict'] == 'PROMOSSA' else 'ko'}">
              {'✓ promossa' if r['verdict'] == 'PROMOSSA' else '✗ respinta'}</span></td>
        </tr>"""
        for _, r in confirmed.iterrows()
    )
    st.html(f"""<table class="ql-table ql-crit">
      <thead><tr><th>Strategia</th><th>Sharpe train</th><th>Sharpe conf.</th>
      <th>Max DD</th><th>Op.</th><th>Criteri</th><th>Esito</th></tr></thead>
      <tbody>{rows}</tbody></table>
      <div class="ql-note">Ordinate per Sharpe sul training, cioè come le ha scelte il sistema.
      Il verdetto usa solo il periodo di conferma.</div>""")

with right, st.container(border=True):
    st.html(card_title("Training → conferma", "quanto si sgonfia lo Sharpe"))
    st.plotly_chart(dumbbell_chart(list(confirmed["rank"]), list(confirmed["train_sharpe"]),
                                   list(confirmed["confirm_sharpe"]), list(confirmed["description"])),
                    config=PLOTLY_CONFIG, width="stretch")
    drop = (confirmed["train_sharpe"] - confirmed["confirm_sharpe"]).mean()
    st.html(f'<div class="ql-note">In media lo Sharpe passa da '
            f'{confirmed["train_sharpe"].mean():.2f} a {confirmed["confirm_sharpe"].mean():.2f} '
            f'({-drop:+.2f}). Le migliori sul training sono in parte migliori per caso.</div>')


# --- Tutte le candidate ----------------------------------------------------------------

with st.container(border=True):
    ok = search[search["status"] == "completed"]
    st.html(card_title("Tutte le candidate", "Sharpe sul training · in blu quelle passate alla conferma"))
    st.plotly_chart(candidates_chart(list(ok["sharpe"]), list(ok["selected"]), list(ok["description"])),
                    config=PLOTLY_CONFIG, width="stretch")
    st.html(f'<div class="ql-note">Mediana {ok["sharpe"].median():.2f}, migliore '
            f'{ok["sharpe"].max():.2f}. Con {len(search)} tentativi, la migliore sembra buona anche '
            'quando nessuna regola ha un vantaggio reale: per questo la conferma è obbligatoria.</div>')


# --- Dettaglio di una candidata ------------------------------------------------------------

@st.cache_data(show_spinner="Calcolo il benchmark…")
def benchmark_returns(universe: tuple[str, ...], start: str, end: str, cost: float) -> pd.Series:
    out = run_experiment(ExperimentSpec(family="benchmark", strategy="buy_and_hold",
                                        params={"rebalance": "M"}, universe=universe,
                                        eval_start=start, eval_end=end, cost_bps=cost))
    return out.result.returns.loc[start:end]


with st.container(border=True):
    st.html(card_title("Dettaglio", "periodo di conferma"))
    choice = st.selectbox("Candidata", confirmed["rank"] + " " + confirmed["description"],
                          label_visibility="collapsed")
    row = confirmed.iloc[list(confirmed["rank"] + " " + confirmed["description"]).index(choice)]
    d_left, d_right = st.columns([1.6, 1], gap="medium")
    with d_left:
        cand_r = registry.returns(row["run_id"])
        bench_r = benchmark_returns(tuple(row["universe"]), row["eval_start"], row["eval_end"],
                                    float(row["cost_bps"]))
        curves = {"Candidata": equity_curve(cand_r, 100), "Pesi uguali": equity_curve(bench_r, 100)}
        st.plotly_chart(equity_chart(curves, color_map(["Candidata", "Pesi uguali"]), log=False),
                        config=PLOTLY_CONFIG, width="stretch")
    with d_right:
        st.html(verdict_block(row["verdict"] == "PROMOSSA", row["description"]))
        st.html(criteria_table([Check(**c) for c in row["checks"]]))


with st.expander(f"Tabella completa delle {len(search)} candidate"):
    table = search.drop(columns=["run_id"]).sort_values("sharpe", ascending=False)
    table[["cagr", "max_drawdown"]] = table[["cagr", "max_drawdown"]].astype(float) * 100
    st.dataframe(table, width="stretch", hide_index=True, column_config={
        "description": "Strategia", "status": "Stato",
        "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
        "cagr": st.column_config.NumberColumn("CAGR", format="%.1f%%"),
        "max_drawdown": st.column_config.NumberColumn("Max DD", format="%.1f%%"),
        "trades": st.column_config.NumberColumn("Operazioni", format="%d"),
        "selected": st.column_config.CheckboxColumn("Confermata"),
    })
