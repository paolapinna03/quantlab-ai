"""Metodi di allocazione: da una matrice di covarianza a pesi di portafoglio.

Tutti i metodi restituiscono pesi long-only che sommano a 1. Tranne la
media-varianza, nessuno usa i rendimenti attesi: le stime dei rendimenti
sono così rumorose che i metodi che ne dipendono tendono a fallire fuori
campione (DeMiguel, Garlappi & Uppal 2009, Review of Financial Studies;
Michaud 1989, Financial Analysts Journal).

Fonti dei metodi:
- Ledoit & Wolf (2004), "A well-conditioned estimator for large-dimensional
  covariance matrices", Journal of Multivariate Analysis 88(2).
- Maillard, Roncalli & Teïletche (2010), "The properties of equally weighted
  risk contribution portfolios", Journal of Portfolio Management 36(4).
- Spinu (2013), formulazione convessa del risk parity (working paper, SSRN).
- López de Prado (2016), "Building diversified portfolios that outperform
  out of sample", Journal of Portfolio Management 42(4): Hierarchical Risk Parity.
- Jagannathan & Ma (2003), Journal of Finance: i vincoli sui pesi
  agiscono come una forma di "shrinkage" e migliorano la minima varianza.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

METHODS = ("inverse_vol", "min_variance", "risk_parity", "hrp", "mean_variance")


# --- Stima della covarianza -------------------------------------------------------

def sample_covariance(returns: np.ndarray) -> np.ndarray:
    x = returns - returns.mean(axis=0)
    return x.T @ x / len(x)


def ledoit_wolf(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Covarianza "ristretta" verso una matrice identità scalata (Ledoit & Wolf 2004).

    Con pochi dati rispetto al numero di asset, la covarianza campionaria è
    instabile; mescolarla con un bersaglio semplice riduce l'errore di stima.
    Restituisce (matrice, intensità di shrinkage tra 0 e 1).
    """
    t, n = returns.shape
    x = returns - returns.mean(axis=0)
    s = x.T @ x / t
    mu = np.trace(s) / n
    target = mu * np.eye(n)
    d2 = np.sum((s - target) ** 2) / n
    b2_bar = sum(np.sum((np.outer(row, row) - s) ** 2) for row in x) / n / t**2
    b2 = min(b2_bar, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    return delta * target + (1 - delta) * s, float(delta)


# --- Metodi di allocazione --------------------------------------------------------

def inverse_vol(cov: np.ndarray) -> np.ndarray:
    """Pesi proporzionali a 1/volatilità: chi oscilla di più pesa meno."""
    w = 1 / np.sqrt(np.diag(cov))
    return w / w.sum()


def min_variance(cov: np.ndarray, max_weight: float = 1.0) -> np.ndarray:
    """Portafoglio di varianza minima, long-only, con tetto per asset."""
    n = len(cov)
    cap = max(max_weight, 1 / n)            # con tetto troppo basso il problema è impossibile
    res = minimize(
        lambda w: w @ cov @ w, np.full(n, 1 / n), jac=lambda w: 2 * cov @ w,
        bounds=[(0.0, cap)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        method="SLSQP", options={"ftol": 1e-12, "maxiter": 500},
    )
    return _clean(res.x)


def risk_parity(cov: np.ndarray) -> np.ndarray:
    """Equal Risk Contribution: ogni asset contribuisce la stessa quota di rischio.

    Formulazione convessa di Spinu: min ½ yᵀΣy − Σ log yᵢ / n, poi w = y / Σy.
    """
    n = len(cov)
    y0 = inverse_vol(cov)
    res = minimize(
        lambda y: 0.5 * y @ cov @ y - np.log(y).sum() / n, y0,
        jac=lambda y: cov @ y - 1 / (n * y), bounds=[(1e-10, None)] * n, method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 1000},
    )
    return _clean(res.x)


def hrp(cov: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity (López de Prado 2016).

    1. raggruppa gli asset per correlazione (clustering gerarchico);
    2. li ordina in modo che i simili stiano vicini;
    3. divide il capitale ricorsivamente tra i gruppi, in proporzione inversa alla loro varianza.
    Non serve invertire la matrice di covarianza, quindi è stabile anche quando è mal condizionata.
    """
    n = len(cov)
    if n == 1:
        return np.ones(1)
    sd = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(sd, sd), -1, 1)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    np.fill_diagonal(dist, 0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="single"))

    def cluster_var(idx):
        sub = cov[np.ix_(idx, idx)]
        w = 1 / np.diag(sub)
        w /= w.sum()
        return w @ sub @ w

    w = np.ones(n)
    clusters = [list(order)]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) < 2:
                continue
            left, right = c[: len(c) // 2], c[len(c) // 2:]
            vl, vr = cluster_var(left), cluster_var(right)
            alpha = 1 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1 - alpha
            nxt += [left, right]
        clusters = nxt
    return _clean(w)


def mean_variance(mu: np.ndarray, cov: np.ndarray, max_weight: float = 1.0) -> np.ndarray:
    """Massimo Sharpe atteso (Markowitz), long-only con tetto.

    Incluso come confronto didattico: usa i rendimenti medi passati come
    stima di quelli futuri, ed è il metodo più sensibile all'errore di stima.
    """
    n = len(cov)
    cap = max(max_weight, 1 / n)
    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu) / vol if vol > 0 else 0.0
    res = minimize(neg_sharpe, np.full(n, 1 / n), bounds=[(0.0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
    return _clean(res.x)


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Quota del rischio totale dovuta a ciascun asset (somma a 1)."""
    marginal = cov @ w
    return w * marginal / (w @ marginal)


def _clean(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.nan_to_num(w, nan=0.0), 0, None)
    w[w < 1e-8] = 0.0
    return w / w.sum()
