"""Stile della dashboard: CSS globale e piccoli blocchi HTML riutilizzabili.

Il tema di base (colori, font, bordi) è in `.streamlit/config.toml`; qui ci
sono solo i ritocchi che il tema non copre.
"""

from __future__ import annotations

import html

MUTED = "#8b9099"
BORDER = "#26292e"

CSS = f"""
<style>
[data-testid="stMainBlockContainer"] {{ padding-top: 2.2rem; padding-bottom: 3rem; }}
[data-testid="stMetricValue"], .ql-num {{ font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }}
[data-testid="stMetricLabel"] p {{
  text-transform: uppercase; letter-spacing: .07em; font-size: .7rem; font-weight: 600; color: {MUTED};
}}
[data-testid="stMetricDelta"] {{ font-size: .78rem; }}

.ql-header {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 16px;
  flex-wrap: wrap; padding-bottom: 14px; border-bottom: 1px solid {BORDER}; margin-bottom: 6px; }}
.ql-eyebrow {{ font-size: .7rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
  color: {MUTED}; margin-bottom: 4px; }}
.ql-title {{ font-size: 1.45rem; font-weight: 600; color: #e8e9eb; margin: 0; line-height: 1.25; }}
.ql-meta {{ font-size: .8rem; color: {MUTED}; margin-top: 6px; }}
.ql-meta b {{ color: #c9ccd1; font-weight: 500; }}
.ql-badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.ql-badge {{ font-size: .72rem; font-weight: 500; padding: 4px 10px; border-radius: 999px;
  border: 1px solid {BORDER}; color: #c9ccd1; background: #17191c; white-space: nowrap; cursor: default; }}
.ql-badge.lock {{ border-color: rgba(57,135,229,.45); color: #8fbaf0; }}
.ql-badge.warn {{ border-color: rgba(201,133,0,.5); color: #e0b45a; }}
.ql-badge.pass {{ border-color: rgba(63,182,139,.55); color: #6fd0a8; font-weight: 600; }}
.ql-badge.fail {{ border-color: rgba(230,103,103,.55); color: #f08f8f; font-weight: 600; }}

.ql-crit td {{ vertical-align: top; }}
.ql-crit .name {{ color: #e8e9eb; font-weight: 500; }}
.ql-crit .why {{ display: block; color: {MUTED}; font-size: .72rem; font-weight: 400;
  margin-top: 2px; line-height: 1.4; max-width: 520px; }}
.ql-status {{ display: inline-block; font-size: .72rem; font-weight: 600; padding: 2px 8px;
  border-radius: 999px; white-space: nowrap; }}
.ql-status.ok {{ color: #6fd0a8; background: rgba(63,182,139,.12); }}
.ql-status.ko {{ color: #f08f8f; background: rgba(230,103,103,.12); }}
.ql-verdict {{ font-size: 1.5rem; font-weight: 700; letter-spacing: .04em; margin: 4px 0 6px; }}
.ql-verdict.ok {{ color: #6fd0a8; }}
.ql-verdict.ko {{ color: #f08f8f; }}
.ql-verdict-text {{ font-size: .8rem; color: #c9ccd1; line-height: 1.5; margin-bottom: 14px; }}

.ql-card-title {{ display: flex; justify-content: space-between; align-items: baseline;
  font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  color: {MUTED}; margin: 2px 0 2px; }}
.ql-card-title span {{ text-transform: none; letter-spacing: 0; font-weight: 400; }}

.ql-table {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
.ql-table th {{ text-align: right; font-weight: 500; color: {MUTED}; font-size: .7rem;
  text-transform: uppercase; letter-spacing: .06em; padding: 6px 4px; border-bottom: 1px solid {BORDER}; }}
.ql-table th:first-child, .ql-table td:first-child {{ text-align: left; }}
.ql-table td {{ text-align: right; padding: 9px 4px; border-bottom: 1px solid {BORDER};
  color: #e8e9eb; font-variant-numeric: tabular-nums; }}
.ql-table tr:last-child td {{ border-bottom: none; }}
.ql-table .se {{ color: {MUTED}; font-size: .74rem; }}
.ql-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 8px;
  vertical-align: 1px; }}
.ql-note {{ font-size: .74rem; color: {MUTED}; margin-top: 10px; line-height: 1.45; }}

.ql-brand {{ font-weight: 700; letter-spacing: .14em; font-size: .9rem; color: #e8e9eb; }}
.ql-brand span {{ color: #3987e5; }}
.ql-sub {{ font-size: .72rem; color: {MUTED}; margin-top: 2px; }}
.ql-section {{ font-size: .68rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
  color: {MUTED}; margin: 18px 0 2px; }}
</style>
"""


def esc(text: str) -> str:
    return html.escape(str(text))


def header(eyebrow: str, title: str, meta: str, badges: list[tuple]) -> str:
    """`meta` può contenere <b>…</b> già sicuri.

    `badges` = [(testo, classe)] oppure [(testo, classe, spiegazione al passaggio del mouse)].
    """
    def chip(b):
        text, cls = b[0], b[1]
        tip = f' title="{esc(b[2])}"' if len(b) > 2 else ""
        return f'<span class="ql-badge {cls}"{tip}>{esc(text)}</span>'
    chips = "".join(chip(b) for b in badges)
    return f"""
<div class="ql-header">
  <div>
    <div class="ql-eyebrow">{esc(eyebrow)}</div>
    <h1 class="ql-title">{esc(title)}</h1>
    <div class="ql-meta">{meta}</div>
  </div>
  <div class="ql-badges">{chips}</div>
</div>"""


def card_title(title: str, hint: str = "") -> str:
    return f'<div class="ql-card-title">{esc(title)}<span>{esc(hint)}</span></div>'


def section(label: str) -> str:
    return f'<div class="ql-section">{esc(label)}</div>'


def _format_check(name: str, value: float) -> str:
    if name == "Max drawdown":
        return f"{value:.1%}"
    if name in ("Deflated Sharpe", "Sharpe > benchmark"):
        return f"{value:.3f}"
    return f"{value:.2f}"


def criteria_table(checks) -> str:
    """Tabella dei criteri: esito con icona + testo, mai solo colore."""
    rows = "".join(
        f"""<tr>
          <td><span class="name">{esc(c.name)}</span><span class="why">{esc(c.meaning)}</span></td>
          <td>{_format_check(c.name, c.value)}</td>
          <td>{esc(c.threshold)}</td>
          <td><span class="ql-status {'ok' if c.passed else 'ko'}">
              {'✓ superato' if c.passed else '✗ fallito'}</span></td>
        </tr>"""
        for c in checks
    )
    return f"""<table class="ql-table ql-crit">
      <thead><tr><th>Criterio</th><th>Valore</th><th>Soglia</th><th>Esito</th></tr></thead>
      <tbody>{rows}</tbody></table>"""


def verdict_block(promoted: bool, text: str, labels: tuple[str, str] = ("PROMOSSA", "RESPINTA")) -> str:
    cls = "ok" if promoted else "ko"
    label = f"✓ {labels[0]}" if promoted else f"✗ {labels[1]}"
    return (f'<div class="ql-verdict {cls}">{label}</div>'
            f'<div class="ql-verdict-text">{esc(text)}</div>')
