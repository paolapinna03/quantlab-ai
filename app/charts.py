"""Grafici Plotly della dashboard.

Regole di stile:
- un colore per entità, sempre lo stesso (SPY è sempre blu, ecc.), mai per rango;
- un solo asse y per grafico;
- linee sottili (2px), griglia discreta, tooltip unificato al passaggio del mouse;
- etichette dirette alla fine delle linee, oltre alla legenda.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

# Palette categoriale validata (daltonismo incluso) sullo sfondo #111214, in ordine fisso.
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SURFACE = "#111214"
TEXT = "#e8e9eb"
TEXT_MUTED = "#8b9099"
GRID = "rgba(232, 233, 235, 0.06)"
FONT = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

PLOTLY_CONFIG = {"displayModeBar": False, "scrollZoom": False}


def color_map(names: list[str]) -> dict[str, str]:
    """Assegna i colori nell'ordine dato: la stessa lista dà sempre gli stessi colori."""
    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(names)}


def _style(fig: go.Figure, height: int, y_format: str, log: bool = False,
           right_margin: int = 8, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=right_margin, t=28 if legend else 8, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT_MUTED, size=11),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#17191c", bordercolor="#26292e",
                        font=dict(family=FONT, color=TEXT, size=12)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left",
                    font=dict(color=TEXT, size=11), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#26292e", ticks="outside", tickcolor="#26292e",
                     showspikes=True, spikecolor=TEXT_MUTED, spikethickness=1,
                     spikedash="dot", spikemode="across")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickformat=y_format,
                     type="log" if log else "linear", side="right" if right_margin < 20 else "left")
    return fig


def equity_chart(curves: dict[str, pd.Series], colors: dict[str, str], log: bool) -> go.Figure:
    """Valore di 100 investito all'inizio della finestra di valutazione."""
    fig = go.Figure()
    for name, s in curves.items():
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name, mode="lines",
            line=dict(color=colors[name], width=2),
            hovertemplate="%{y:,.1f}",
        ))
    height = 380
    _style(fig, height, ",.0f", log, right_margin=150)
    x_end = max(s.index[-1] for s in curves.values())
    fig.update_xaxes(range=[min(s.index[0] for s in curves.values()), x_end])

    # Etichette a fine linea, distanziate per non sovrapporsi.
    to_axis = math.log10 if log else (lambda v: v)
    values = {n: s.dropna().iloc[-1] for n, s in curves.items()}
    lo = to_axis(min(s.min() for s in curves.values()))
    hi = to_axis(max(s.max() for s in curves.values()))
    min_gap = (hi - lo) * 16 / (height - 60)  # ~16 px in unità dell'asse
    placed = _spread({n: to_axis(v) for n, v in values.items()}, min_gap)
    for name, y in placed.items():
        fig.add_annotation(
            x=x_end, y=y, text=f"{name}  <b>{values[name]:,.0f}</b>",
            xanchor="left", xshift=8, showarrow=False, font=dict(color=TEXT, size=11),
        )
    return fig


def _spread(positions: dict[str, float], min_gap: float) -> dict[str, float]:
    """Sposta le etichette verticalmente finché distano almeno `min_gap`."""
    ordered = sorted(positions.items(), key=lambda kv: kv[1])
    out: dict[str, float] = {}
    prev = -math.inf
    for name, y in ordered:
        y = max(y, prev + min_gap)
        out[name] = y
        prev = y
    return out


def drawdown_chart(drawdowns: dict[str, pd.Series], colors: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    for name, s in drawdowns.items():
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name, mode="lines",
            line=dict(color=colors[name], width=1.5),
            hovertemplate="%{y:.1%}",
        ))
    fig.add_hline(y=0, line=dict(color="#26292e", width=1))
    fig = _style(fig, 220, ".0%", legend=False)
    first = next(iter(drawdowns.values()))
    fig.update_xaxes(range=[first.index[0], first.index[-1]])
    return fig


def weights_chart(weights: pd.DataFrame, colors: dict[str, str], height: int = 220) -> go.Figure:
    """Pesi detenuti nel tempo (area impilata); lo spazio sopra è liquidità."""
    fig = go.Figure()
    for asset in weights.columns:
        fig.add_trace(go.Scatter(
            x=weights.index, y=weights[asset].values, name=asset, mode="lines",
            stackgroup="w", line=dict(width=0),
            fillcolor=colors[asset], hovertemplate="%{y:.0%}",
        ))
    fig = _style(fig, height, ".0%")
    fig.update_layout(legend=dict(font=dict(size=10), itemwidth=30, tracegroupgap=0))
    fig.update_yaxes(range=[0, 1.0], dtick=0.25)
    fig.update_xaxes(range=[weights.index[0], weights.index[-1]])
    return fig


def walk_forward_chart(folds: pd.DataFrame, height: int = 230) -> go.Figure:
    """Sharpe di ogni anno di test, con il lookback scelto usando solo gli anni precedenti."""
    years = [str(pd.Timestamp(d).year) for d in folds["test_start"]]
    months = [f"{p.get('lookback', 0) // 21}m" for p in folds["params"]]
    fig = go.Figure(go.Bar(
        x=years, y=folds["test_sharpe_chosen"], marker=dict(color=PALETTE[0], cornerradius=3),
        customdata=months,
        hovertemplate="Anno %{x}<br>Lookback %{customdata}<br>Sharpe %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#3a3e45", width=1))
    fig = _style(fig, height, ".1f", legend=False)
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_xaxes(showspikes=False, tickfont=dict(size=10))
    return fig


def dumbbell_chart(labels: list[str], train: list[float], confirm: list[float],
                   hovers: list[str], height: int = 260) -> go.Figure:
    """Sharpe in training (scelto) contro Sharpe in conferma (mai visto): quanto si sgonfia."""
    fig = go.Figure()
    for lab, a, b in zip(labels, train, confirm):
        fig.add_trace(go.Scatter(x=[a, b], y=[lab, lab], mode="lines",
                                 line=dict(color="#3a3e45", width=2), showlegend=False,
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=train, y=labels, mode="markers", name="Training (2005–2016)",
                             marker=dict(color=TEXT_MUTED, size=10, line=dict(color=SURFACE, width=2)),
                             customdata=hovers,
                             hovertemplate="%{customdata}<br>Training: %{x:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=confirm, y=labels, mode="markers", name="Conferma (2017–2022)",
                             marker=dict(color=PALETTE[0], size=10, line=dict(color=SURFACE, width=2)),
                             customdata=hovers,
                             hovertemplate="%{customdata}<br>Conferma: %{x:.2f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#3a3e45", width=1))
    fig = _style(fig, height, ".1f")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(showgrid=True, gridcolor=GRID, showspikes=False, title=dict(
        text="Sharpe ratio", font=dict(size=10, color=TEXT_MUTED)))
    fig.update_yaxes(autorange="reversed", showgrid=False, side="left", tickformat="")
    return fig


def candidates_chart(sharpes: list[float], selected: list[bool], hovers: list[str],
                     height: int = 230) -> go.Figure:
    """Sharpe in training di tutte le candidate, ordinate; in evidenza quelle scelte."""
    order = sorted(range(len(sharpes)), key=lambda i: sharpes[i], reverse=True)
    colors = [PALETTE[0] if selected[i] else "#3a3e45" for i in order]
    fig = go.Figure(go.Bar(
        x=list(range(1, len(order) + 1)), y=[sharpes[i] for i in order],
        marker=dict(color=colors, cornerradius=2), customdata=[hovers[i] for i in order],
        hovertemplate="%{customdata}<br>Sharpe training %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#3a3e45", width=1))
    fig = _style(fig, height, ".1f", legend=False)
    fig.update_layout(hovermode="closest", bargap=0.25)
    fig.update_xaxes(showspikes=False, title=dict(text="candidate, dalla migliore alla peggiore",
                                                  font=dict(size=10, color=TEXT_MUTED)))
    return fig
