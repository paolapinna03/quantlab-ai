"""Grafici Plotly della dashboard.

Regole di stile:
- un colore per entità, sempre lo stesso (SPY è sempre blu, ecc.), mai per rango;
- un solo asse y per grafico;
- linee sottili (2px), griglia discreta, tooltip unificato al passaggio del mouse;
- etichette dirette alla fine delle linee, oltre alla legenda.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Palette categoriale validata per sfondo scuro (daltonismo incluso), in ordine fisso.
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SURFACE = "#1a1a19"
TEXT = "#f0efec"
TEXT_MUTED = "#c3c2b7"
GRID = "rgba(240, 239, 236, 0.08)"


def color_map(names: list[str]) -> dict[str, str]:
    """Assegna i colori nell'ordine dato: la stessa lista dà sempre gli stessi colori."""
    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(names)}


def _style(fig: go.Figure, height: int, y_format: str, log: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=96, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, size=12),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE, font_color=TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=TEXT)),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, showspikes=True, spikecolor=TEXT_MUTED,
                     spikethickness=1, spikedash="dot", spikemode="across")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickformat=y_format,
                     type="log" if log else "linear")
    return fig


def _end_labels(fig: go.Figure, series: dict[str, pd.Series], fmt: str) -> None:
    for name, s in series.items():
        s = s.dropna()
        if s.empty:
            continue
        fig.add_annotation(
            x=s.index[-1], y=s.iloc[-1], text=f"{name}: {s.iloc[-1]:{fmt}}",
            xanchor="left", xshift=6, showarrow=False, font=dict(color=TEXT, size=11),
            yref="y",
        )


def equity_chart(curves: dict[str, pd.Series], colors: dict[str, str], log: bool) -> go.Figure:
    """Valore di 100 investito all'inizio della finestra di valutazione."""
    fig = go.Figure()
    for name, s in curves.items():
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name, mode="lines",
            line=dict(color=colors[name], width=2),
            hovertemplate="%{y:,.1f}",
        ))
    _style(fig, 420, ",.0f", log)
    if not log:
        _end_labels(fig, curves, ",.0f")
    return fig


def drawdown_chart(drawdowns: dict[str, pd.Series], colors: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    for name, s in drawdowns.items():
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name, mode="lines",
            line=dict(color=colors[name], width=2),
            hovertemplate="%{y:.1%}",
        ))
    fig.add_hline(y=0, line=dict(color=GRID, width=1))
    return _style(fig, 260, ".0%")


def weights_chart(weights: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    """Pesi detenuti nel tempo (area impilata); lo spazio sopra è liquidità."""
    fig = go.Figure()
    for asset in weights.columns:
        fig.add_trace(go.Scatter(
            x=weights.index, y=weights[asset].values, name=asset, mode="lines",
            stackgroup="w", line=dict(color=SURFACE, width=0.5),
            fillcolor=colors[asset], hovertemplate="%{y:.0%}",
        ))
    fig = _style(fig, 300, ".0%")
    fig.update_yaxes(range=[0, 1.02])
    return fig
