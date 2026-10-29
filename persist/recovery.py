"""Parameter recovery, model recovery and power analysis for the proposed
within-subject design (five blocks).

A simulated participant contributes one quit time per block. The exact
likelihood of a quit time q (0..T) under a model with parameters (c, kappa, tau)
is  prod_{t<q} sigma(z_t) * [1 - sigma(z_q)]^{q<T},  z_t = (U_t - c)/tau, where
the decision variable U_t = V_t + kappa*S_t is a deterministic function of the
stream the participant actually experienced.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

from . import core, experiments as ex

BLOCKS = ["CRF_EXT", "P50_EXT", "CUE_REV", "DEV_P50", "DEV_CUE_N20"]
MODELS_FIT = list(core.MODELS)                     # includes the null model "N"
INTRINSIC = set(core.INTRINSIC_MODELS)
NPAR = {"N": 1, "R": 2, "CD": 2, "U": 3, "Sp": 3, "IG": 3, "LPs": 3, "LPa": 3, "H": 3}
GRID = dict(c=np.linspace(0.03, 0.36, 12), rho=np.array([0.125, 0.25, 0.5, 1.0, 2.0]),
            tau=np.array([0.005, 0.015, 0.04]), q=np.logspace(np.log10(0.003), np.log10(0.1), 25))
TRUE_RANGE = dict(c=(0.10, 0.30), rho=(0.25, 2.0), tau=(0.005, 0.03), q=(0.005, 0.05))


def build_pool(cfg, n_pool, seed, blocks=BLOCKS):
    """Per block: signals (V, S) of every non-null model for n_pool streams."""
    cells = ex.build_cells(cfg)
    pool = {}
    for b in blocks:
        sim = ex.simulate_cell(cells[b], cfg, n_pool, seed)
        pool[b] = {m: core.model_VS(m, sim["leaky"], sim["boc"], pay=sim["pay"])
                   for m in MODELS_FIT if m != "N"}
    return pool


def _loguniform(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def sample_participants(model, n, rng):
    if model == "N":
        return dict(q=_loguniform(rng, *TRUE_RANGE["q"], n))
    p = dict(c=rng.uniform(*TRUE_RANGE["c"], n), tau=_loguniform(rng, *TRUE_RANGE["tau"], n))
    if model in INTRINSIC:
        p["rho"] = _loguniform(rng, *TRUE_RANGE["rho"], n)
    return p


def generate_quit_times(model, parts, pool, idx, units, T, rng, blocks=BLOCKS):
    """Quit times (n, n_blocks) for simulated participants. idx: (n, n_blocks)
    indices of the stream each participant experienced in each block."""
    n = idx.shape[0]
    Q = np.zeros((n, len(blocks)), dtype=int)
    for j, b in enumerate(blocks):
        v = rng.random((n, T))
        if model == "N":
            Q[:, j] = np.where((v < parts["q"][:, None]).any(1), (v < parts["q"][:, None]).argmax(1), T)
            continue
        V, S = pool[b][model]
        kappa = parts["rho"] * units[model] if model in INTRINSIC else np.zeros(n)
        U = V[idx[:, j], :T] + kappa[:, None] * S[idx[:, j], :T]
        engage = v < expit((U - parts["c"][:, None]) / parts["tau"][:, None])
        dis = ~engage
        Q[:, j] = np.where(dis.any(1), dis.argmax(1), T)
    return Q


def fit_model(model, pool, idx, Q, units, T, chunk=30, blocks=BLOCKS):
    """Maximum log-likelihood over the parameter grid for every participant.
    Returns (ll_max (n,), best parameter dict)."""
    n = Q.shape[0]
    if model == "N":
        qs = GRID["q"]
        ll = np.zeros((n, len(qs)))
        for j in range(len(blocks)):
            q = Q[:, j][:, None]
            ll += q * np.log1p(-qs)[None, :] + (q < T) * np.log(qs)[None, :]
        k = ll.argmax(1)
        return ll.max(1), dict(q=qs[k])
    cs, taus = GRID["c"], GRID["tau"]
    kap = GRID["rho"] * units[model] if model in INTRINSIC else np.array([0.0])
    nk, nc, nt = len(kap), len(cs), len(taus)
    LL = np.zeros((n, nk, nc, nt))
    for j, b in enumerate(blocks):
        V, S = pool[b][model]
        for a in range(0, n, chunk):
            z_ = slice(a, min(n, a + chunk))
            ids = idx[z_, j]
            m = len(ids)
            U = V[ids, :T][:, None, :] + kap[None, :, None] * S[ids, :T][:, None, :]      # (m, nk, T)
            z = (U[:, :, None, None, :] - cs[None, None, :, None, None]) / taus[None, None, None, :, None]
            lsig = -np.logaddexp(0.0, -z)
            lns = -np.logaddexp(0.0, z)
            cum = np.cumsum(lsig, axis=-1)
            q = Q[z_, j]
            qm1 = np.clip(q - 1, 0, T - 1)[:, None, None, None, None]
            qc = np.clip(q, 0, T - 1)[:, None, None, None, None]
            cq = np.take_along_axis(cum, np.broadcast_to(qm1, (m, nk, nc, nt, 1)), -1)[..., 0]
            lq = np.take_along_axis(lns, np.broadcast_to(qc, (m, nk, nc, nt, 1)), -1)[..., 0]
            qq = q[:, None, None, None]
            LL[z_] += np.where(qq > 0, cq, 0.0) + np.where(qq < T, lq, 0.0)
    flat = LL.reshape(n, -1)
    k = flat.argmax(1)
    ik, ic, it = np.unravel_index(k, (nk, nc, nt))
    best = dict(c=cs[ic], tau=taus[it])
    if model in INTRINSIC:
        best["rho"] = GRID["rho"][ik]
    return flat.max(1), best


def fit_all(pool, idx, Q, units, T, models=MODELS_FIT, chunk=30):
    ll = np.zeros((Q.shape[0], len(models)))
    best = {}
    for i, m in enumerate(models):
        ll[:, i], best[m] = fit_model(m, pool, idx, Q, units, T, chunk=chunk)
    return ll, best


def group_winner(ll, rows, models=MODELS_FIT, criterion="aic"):
    """Model with the lowest group criterion for the participants in `rows`."""
    k = np.array([NPAR[m] for m in models], dtype=float)
    pen = 2.0 * k if criterion == "aic" else k * np.log(len(BLOCKS))
    score = -2.0 * ll[rows].sum(0) + len(rows) * pen
    return int(score.argmin())


def power_curve(ll_by_true, ns, n_rep=300, seed=0, models=MODELS_FIT, criterion="aic"):
    """P(correct group-level identification) as a function of sample size."""
    rng = np.random.default_rng(seed)
    out = {}
    for true, ll in ll_by_true.items():
        t = models.index(true)
        total = ll.shape[0]
        for n in ns:
            if n > total:
                continue
            hit = 0
            for _ in range(n_rep):
                rows = rng.choice(total, n, replace=False)
                hit += group_winner(ll, rows, models, criterion) == t
            out[(true, n)] = hit / n_rep
    return out


def confusion_matrix(ll_by_true, n, n_rep=300, seed=1, models=MODELS_FIT, criterion="aic"):
    rng = np.random.default_rng(seed)
    C = np.zeros((len(models), len(models)))
    for true, ll in ll_by_true.items():
        t = models.index(true)
        for _ in range(n_rep):
            rows = rng.choice(ll.shape[0], n, replace=False)
            C[t, group_winner(ll, rows, models, criterion)] += 1
    return C / n_rep


def parameter_recovery(model, parts, best):
    out = {}
    for key in ("c", "rho", "tau", "q"):
        if key in parts:
            r = spearmanr(parts[key], best[key])[0]
            out[key] = float(r)
    return out
