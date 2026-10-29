"""Core simulation code for the persistence-after-withdrawal study.

Design
------
A focal activity is engaged on every trial of an acquisition phase (forced
engagement, so every agent receives an identical observation stream), then in a
test phase the agent decides trial by trial whether to keep engaging. The first
disengagement is absorbing ("quit"). Persistence = number of engaged test trials
before the first quit (right-censored at n_test).

Because the observation stream does not depend on the agent's actions while it
is engaged, an agent's decision variable U_k (after k test observations) can be
computed once for the whole stream; the quit time then follows from U, a
threshold c, decision noise tau and pre-drawn uniforms v (common random numbers
across agents).
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma, expit

LN2 = float(np.log(2.0))
MODELS = ("N", "R", "CD", "U", "Sp", "IG", "LPs", "LPa", "H")
INTRINSIC_MODELS = ("U", "Sp", "IG", "LPs", "LPa", "H")


# --------------------------------------------------------------------------
# Task streams
# --------------------------------------------------------------------------
def phase_probs(kind: str, p: float = 0.5, eps: float = 0.05):
    """Return (P(reward|cue 0), P(reward|cue 1)) for a named schedule."""
    if kind == "crf":
        return (1.0, 1.0)
    if kind == "random":
        return (float(p), float(p))
    if kind == "cued":
        return (1.0 - eps, eps)
    if kind == "cued_rev":
        return (eps, 1.0 - eps)
    if kind == "ext":
        return (0.0, 0.0)
    raise ValueError(f"unknown schedule kind: {kind}")


def draw_noise(n_seeds: int, n_acq: int, n_test: int, seed: int):
    """Pre-draw every random number a replication block needs.

    cue : (S, L) int8, fair coin; u : (S, L) outcome uniforms;
    v : (S, n_test) decision-noise uniforms. L = n_acq + n_test.
    """
    rng = np.random.default_rng(seed)
    length = n_acq + n_test
    cue = (rng.random((n_seeds, length)) < 0.5).astype(np.int8)
    u = rng.random((n_seeds, length))
    v = rng.random((n_seeds, n_test))
    return cue, u, v


def make_outcomes(cue, u, n_acq, acq, test):
    """Binary outcomes from cue-conditional reward probabilities."""
    p_acq = np.where(cue[:, :n_acq] == 0, acq[0], acq[1])
    p_tst = np.where(cue[:, n_acq:] == 0, test[0], test[1])
    return (u < np.concatenate([p_acq, p_tst], axis=1)).astype(np.int8)


def binary_entropy_bits(p):
    """Entropy (bits) of a Bernoulli(p) outcome."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


# --------------------------------------------------------------------------
# Information gain for a Bernoulli parameter with a Beta posterior
# --------------------------------------------------------------------------
def bald_bernoulli(a, b):
    """Expected information gain (nats) about p ~ Beta(a, b) from one Bernoulli
    observation: I(p; o) = H(E[p]) - E[H(p)] (closed form via digamma)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = a / (a + b)
    h_mean = -(m * np.log(m) + (1.0 - m) * np.log1p(-m))
    e_h = digamma(a + b + 1.0) - m * digamma(a + 1.0) - (1.0 - m) * digamma(b + 1.0)
    return h_mean - e_h


def lp_from_losses(loss, n, w):
    """Signed learning progress after n observations: mean log-loss in the older
    window minus mean log-loss in the most recent window (positive = loss falling).
    Zero until 2*w observations exist."""
    if n < 2 * w:
        return np.zeros(loss.shape[0])
    return loss[:, n - 2 * w:n - w].mean(1) - loss[:, n - w:n].mean(1)


def lp_traj(loss, n_acq, n_test, w):
    """Signed learning progress (units of ln 2) after k = 0..n_test test
    observations, recomputed from a loss history for any window w."""
    K = n_test + 1
    out = np.zeros((loss.shape[0], K))
    for k in range(K):
        out[:, k] = lp_from_losses(loss, n_acq + k, w)
    return out / LN2


# --------------------------------------------------------------------------
# Shared leaky Beta-Bernoulli world model (agents R, U, IG, LPs, LPa)
# --------------------------------------------------------------------------
def leaky_signals(cue, o, n_acq, n_test, lam=0.95, a0=0.5, b0=0.5, w=10, eta=0.1,
                  assoc0=0.5):
    """Run the leaky per-cue Beta-Bernoulli model over the whole stream.

    Returns arrays of shape (S, n_test + 1); column k is the agent state after k
    test observations (k = 0 is the end of acquisition).
      V     reward expectation, averaged over the two equiprobable cues
      ent   outcome (aleatoric) uncertainty: predictive entropy in bits, cue-averaged
      assoc surprise: running mean of |prediction error| (Pearce-Hall-style associability)
      ig    expected information gain, normalized by its value at the prior
      lp    signed learning progress (older-window minus recent-window mean
            log-loss), in units of ln 2
    """
    S, L = o.shape
    A = np.zeros((S, 2))
    B = np.zeros((S, 2))
    idx = np.arange(S)
    loss = np.zeros((S, L))
    assoc = np.full(S, assoc0)
    K = n_test + 1
    V = np.zeros((S, K))
    AS = np.zeros((S, K))
    EN = np.zeros((S, K))
    IG = np.zeros((S, K))
    LP = np.zeros((S, K))
    ig0 = float(bald_bernoulli(a0, b0))
    for t in range(L):
        s = cue[:, t].astype(int)
        y = o[:, t].astype(float)
        mu = (a0 + A[idx, s]) / (a0 + b0 + A[idx, s] + B[idx, s])
        loss[:, t] = -np.log(np.where(y == 1.0, mu, 1.0 - mu))
        assoc = (1.0 - eta) * assoc + eta * np.abs(y - mu)
        A[idx, s] = lam * A[idx, s] + y
        B[idx, s] = lam * B[idx, s] + (1.0 - y)
        k = t - (n_acq - 1)
        if k >= 0:
            mean_c = (a0 + A) / (a0 + b0 + A + B)
            V[:, k] = mean_c.mean(axis=1)
            EN[:, k] = binary_entropy_bits(mean_c).mean(axis=1)
            AS[:, k] = assoc
            IG[:, k] = bald_bernoulli(a0 + A, b0 + B).mean(axis=1) / ig0
            LP[:, k] = lp_from_losses(loss, t + 1, w)
    return dict(V=V, assoc=AS, ent=EN, ig=IG, lp=LP / LN2, loss=loss)


# --------------------------------------------------------------------------
# Bayesian online change-point model (agents CD and H)
# --------------------------------------------------------------------------
def bocpd_signals(cue, o, n_acq, n_test, h=0.02, a0=0.5, b0=0.5, w=10):
    """Bayesian online change-point detection (Adams & MacKay, 2007) with
    per-cue Beta-Bernoulli run models and a constant hazard h.

    Returns V (expected next-trial reward under the run-length posterior,
    cue-averaged), the signed learning progress of the model's own predictive
    log-loss, and pcp, the posterior probability that the current run is
    shorter than w observations (an index of detected change).
    """
    S, L = o.shape
    R = L + 2
    A = np.zeros((S, R, 2))
    B = np.zeros((S, R, 2))
    P = np.zeros((S, R))
    P[:, 0] = 1.0
    idx = np.arange(S)
    loss = np.zeros((S, L))
    K = n_test + 1
    V = np.zeros((S, K))
    LP = np.zeros((S, K))
    PCP = np.zeros((S, K))
    for t in range(L):
        m = t + 1
        s = cue[:, t].astype(int)
        y = o[:, t].astype(float)
        a = a0 + A[idx, :m, s]
        b = b0 + B[idx, :m, s]
        mu = a / (a + b)
        lik = np.where(y[:, None] == 1.0, mu, 1.0 - mu)
        joint = P[:, :m] * lik
        pred = joint.sum(axis=1)
        loss[:, t] = -np.log(pred)
        newP = np.empty((S, m + 1))
        newP[:, 0] = h * pred
        newP[:, 1:] = (1.0 - h) * joint
        newP /= pred[:, None]
        P[:, :m + 1] = newP
        A[:, 1:m + 1] = A[:, :m].copy()
        B[:, 1:m + 1] = B[:, :m].copy()
        A[:, 0] = 0.0
        B[:, 0] = 0.0
        A[idx, 1:m + 1, s] += y[:, None]
        B[idx, 1:m + 1, s] += (1.0 - y)[:, None]
        k = t - (n_acq - 1)
        if k >= 0:
            mm = m + 1
            mean_r = ((a0 + A[:, :mm]) / (a0 + b0 + A[:, :mm] + B[:, :mm])).mean(axis=2)
            V[:, k] = (P[:, :mm] * mean_r).sum(axis=1)
            PCP[:, k] = P[:, :min(w, mm)].sum(axis=1)
            LP[:, k] = lp_from_losses(loss, t + 1, w)
    return dict(V=V, lp=LP / LN2, pcp=PCP, loss=loss)


# --------------------------------------------------------------------------
# Models -> (V, S) pairs; decision variable is U = V + kappa * S
# --------------------------------------------------------------------------
def model_VS(model: str, leaky: dict, boc: dict, pay: float = 1.0, lp_override=None):
    """Extrinsic term V and intrinsic signal S (both (S, K)) for a model.

    `pay` multiplies the extrinsic term (0 = payment withdrawn while outcomes
    continue to occur and remain informative). `lp_override` = (leaky_lp, boc_lp)
    substitutes learning-progress trajectories computed with another window."""
    lk_lp = leaky["lp"] if lp_override is None else lp_override[0]
    bc_lp = None if boc is None else (boc["lp"] if lp_override is None else lp_override[1])
    if model == "R":
        V, S = leaky["V"], np.zeros_like(leaky["V"])
    elif model == "CD":
        V, S = boc["V"], np.zeros_like(boc["V"])
    elif model == "U":
        V, S = leaky["V"], leaky["ent"]
    elif model == "Sp":
        V, S = leaky["V"], leaky["assoc"]
    elif model == "IG":
        V, S = leaky["V"], leaky["ig"]
    elif model == "LPs":
        V, S = leaky["V"], lk_lp
    elif model == "LPa":
        V, S = leaky["V"], np.abs(lk_lp)
    elif model == "LPp":      # positive part: only improvement counts (sensitivity analysis)
        V, S = leaky["V"], np.maximum(lk_lp, 0.0)
    elif model == "H":
        V, S = boc["V"], np.abs(bc_lp)
    else:
        raise ValueError(model)
    return V * pay, S


# --------------------------------------------------------------------------
# Decision rule
# --------------------------------------------------------------------------
def quit_times(U, v, c, tau):
    """Number of engaged test trials before the first disengagement.

    U : (S, K>=n_test) decision variable before each test trial;
    v : (S, n_test) uniforms. Engage on trial j iff v[:, j] < sigmoid((U-c)/tau).
    """
    n_test = v.shape[1]
    Ux = U[:, :n_test]
    if tau <= 0:
        engage = Ux > c
    else:
        engage = v < expit((Ux - c) / tau)
    dis = ~engage
    anyq = dis.any(axis=1)
    return np.where(anyq, dis.argmax(axis=1), n_test)


def quit_times_null(v, q):
    """Null model: constant per-trial quit probability q (no experience dependence)."""
    n_test = v.shape[1]
    dis = v < q
    return np.where(dis.any(axis=1), dis.argmax(axis=1), n_test)


def calibrate_threshold(U, v, tau, target, lo=-1.0, hi=3.0, iters=60):
    """Find c so that mean persistence in the reference cell equals `target`.

    Mean persistence is non-increasing in c, so bisection applies. Returns
    (c, achieved_mean, status).
    """
    f = lambda c: quit_times(U, v, c, tau).mean()
    f_lo, f_hi = f(lo), f(hi)
    if not (f_hi <= target <= f_lo):
        return float("nan"), float("nan"), "target_not_bracketed"
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    return c, float(f(c)), "ok"


def calibrate_null(v, target, iters=60):
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if quit_times_null(v, mid).mean() > target:
            lo = mid
        else:
            hi = mid
    q = 0.5 * (lo + hi)
    return q, float(quit_times_null(v, q).mean())


def equal_magnitude_kappa(V, S, window=20, rho=0.5):
    """Weight kappa such that the mean absolute intrinsic contribution kappa*|S|
    equals rho times the mean absolute extrinsic term |V|, both averaged over the
    first `window` test trials of the reference cell. rho = 0.5 is the default
    ("the intrinsic term is worth half the extrinsic term, on average");
    sensitivity analyses vary rho over a 16-fold range."""
    mv = np.abs(V[:, :window]).mean()
    ms = np.abs(S[:, :window]).mean()
    return float(rho * mv / ms) if ms > 1e-9 else 0.0
