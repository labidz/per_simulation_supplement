"""Experiment definitions, calibration and evaluation."""
from __future__ import annotations

import numpy as np

from . import core

DEFAULT_CFG = dict(
    T=300, N_main=120, N_short=20, cue_eps=0.05, lean_p=0.10,
    lam=0.95, a0=0.5, b0=0.5, h=0.02, w=10, eta=0.1, tau=0.01,
    target=40.0, kappa_window=20, rho=0.5, ref_cell="P50_EXT",
    cal_seeds=2000, cal_seed=111, eval_seeds=1000, eval_seed=20260101,
)


def build_cells(cfg):
    N, eps, lean = cfg["N_main"], cfg["cue_eps"], cfg["lean_p"]
    N_short = cfg["N_short"]
    ext = ("ext", {})
    cells = {}

    def add(name, n_acq, acq, test, pay=1.0):
        cells[name] = dict(n_acq=n_acq, acq=acq, test=test, pay=pay)

    cued, crf, rnd = ("cued", {"eps": eps}), ("crf", {}), ("random", {"p": 0.5})
    # E1 sanity: reward continues unchanged
    add("CRF_CONT", N, crf, crf)
    add("P50_CONT", N, rnd, rnd)
    add("CUE_CONT", N, cued, cued)
    # E2/E3 extinction after reliable, partial, and cue-predictable reward
    add("CRF_EXT", N, crf, ext)
    for p in (0.75, 0.5, 0.25):
        add(f"P{int(round(p * 100))}_EXT", N, ("random", {"p": p}), ext)
    add("CUE_EXT", N, cued, ext)
    # E4 degraded (lean, noisy) reward instead of none
    for nm, acq in (("CRF", crf), ("P50", rnd), ("CUE", cued)):
        add(f"{nm}_LEAN", N, acq, ("random", {"p": lean}))
    # E5 reversal of the cue-outcome mapping with reward rate unchanged
    add("CUE_REV", N, cued, ("cued_rev", {"eps": eps}))
    # E6 payment withdrawn, outcomes keep occurring and stay informative
    add("DEV_CRF", N, crf, crf, pay=0.0)
    add("DEV_P50", N, rnd, rnd, pay=0.0)
    add("DEV_CUE", N, cued, cued, pay=0.0)
    add("DEV_CUE_N20", N_short, cued, cued, pay=0.0)
    add("DEV_CUEREV", N, cued, ("cued_rev", {"eps": eps}), pay=0.0)
    return cells


def simulate_cell(cell, cfg, n_seeds, base_seed, with_boc=True):
    n_acq, T = cell["n_acq"], cfg["T"]
    cue, u, v = core.draw_noise(n_seeds, n_acq, T, base_seed + n_acq)
    acq = core.phase_probs(cell["acq"][0], **cell["acq"][1])
    tst = core.phase_probs(cell["test"][0], **cell["test"][1])
    o = core.make_outcomes(cue, u, n_acq, acq, tst)
    leaky = core.leaky_signals(cue, o, n_acq, T, lam=cfg["lam"], a0=cfg["a0"], b0=cfg["b0"],
                               w=cfg["w"], eta=cfg["eta"])
    boc = core.bocpd_signals(cue, o, n_acq, T, h=cfg["h"], a0=cfg["a0"], b0=cfg["b0"],
                             w=cfg["w"]) if with_boc else None
    return dict(leaky=leaky, boc=boc, v=v, o=o, cue=cue, n_acq=n_acq, pay=cell.get("pay", 1.0))


def calibrate_models(cfg, cells=None, models=core.MODELS, sim=None):
    """Fix the shared engagement cost c on the reference cell (so that the
    conventional reward-only agent R persists `target` trials on average), the
    null hazard q likewise, and each intrinsic weight kappa by the equal-magnitude
    rule. All agents share c and tau; calibration seeds are never reused for
    evaluation."""
    cells = cells or build_cells(cfg)
    if sim is None:
        sim = simulate_cell(cells[cfg["ref_cell"]], cfg, cfg["cal_seeds"], cfg["cal_seed"])
    V_r, _ = core.model_VS("R", sim["leaky"], sim["boc"])
    c_task, got, status = core.calibrate_threshold(V_r, sim["v"], cfg["tau"], cfg["target"])
    params = {}
    for m in models:
        if m == "N":
            q, got_n = core.calibrate_null(sim["v"], cfg["target"])
            params["N"] = dict(q=q, achieved=got_n, status="ok")
            continue
        V, Sg = core.model_VS(m, sim["leaky"], sim["boc"])
        kappa = 0.0 if m in ("R", "CD") else core.equal_magnitude_kappa(
            V, Sg, cfg["kappa_window"], cfg["rho"])
        U = V + kappa * Sg
        params[m] = dict(c=c_task, kappa=kappa, status=status,
                         ref_persistence=float(core.quit_times(U, sim["v"], c_task, cfg["tau"]).mean()))
    return params


def persistence_for(model, params, sim, cfg):
    if model == "N":
        return core.quit_times_null(sim["v"], params["N"]["q"])
    V, Sg = core.model_VS(model, sim["leaky"], sim["boc"], pay=sim.get("pay", 1.0))
    U = V + params[model]["kappa"] * Sg
    return core.quit_times(U, sim["v"], params[model]["c"], cfg["tau"])


def run_cells(cfg, params, cells, names=None, n_seeds=None, base_seed=None, models=core.MODELS,
              keep_traj=False):
    """Evaluate all models on the requested cells. Returns persistence arrays."""
    n_seeds = n_seeds or cfg["eval_seeds"]
    base_seed = base_seed if base_seed is not None else cfg["eval_seed"]
    names = names or list(cells)
    out = {m: {} for m in models}
    traj = {m: {} for m in models} if keep_traj else None
    for name in names:
        sim = simulate_cell(cells[name], cfg, n_seeds, base_seed)
        for m in models:
            out[m][name] = persistence_for(m, params, sim, cfg)
            if keep_traj and m != "N":
                V, Sg = core.model_VS(m, sim["leaky"], sim["boc"], pay=sim.get("pay", 1.0))
                U = V + params[m]["kappa"] * Sg
                traj[m][name] = dict(U=U.mean(0), V=V.mean(0), S=Sg.mean(0))
    return (out, traj) if keep_traj else out


# -------------------------------------------------------------------- statistics
def paired_contrast(qa, qb, n_boot=2000, seed=0):
    """Mean paired difference qa - qb with a seed-level bootstrap 95% CI,
    standardized effect d_z and probability of superiority."""
    d = (qa - qb).astype(float)
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    sd = d.std(ddof=1)
    return dict(diff=float(d.mean()), lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)),
                dz=float(d.mean() / sd) if sd > 0 else 0.0,
                p_sup=float((qa > qb).mean() + 0.5 * (qa == qb).mean()))


def sign_label(res, margin=0.0):
    """'+' / '-' if the CI excludes +/- margin, else '0'."""
    if res["lo"] > margin:
        return "+"
    if res["hi"] < -margin:
        return "-"
    return "0"


def set_window(sim, cfg, w):
    """Copy of `sim` whose learning-progress trajectories use window w."""
    out = dict(sim)
    T = cfg["T"]
    out["leaky"] = dict(sim["leaky"], lp=core.lp_traj(sim["leaky"]["loss"], sim["n_acq"], T, w))
    if sim["boc"] is not None:
        out["boc"] = dict(sim["boc"], lp=core.lp_traj(sim["boc"]["loss"], sim["n_acq"], T, w))
    return out


def resimulate_leaky(sim, cfg):
    """Recompute only the leaky-model signals (new lam/eta/w) on the same stream."""
    out = dict(sim)
    out["leaky"] = core.leaky_signals(sim["cue"], sim["o"], sim["n_acq"], cfg["T"], lam=cfg["lam"],
                                      a0=cfg["a0"], b0=cfg["b0"], w=cfg["w"], eta=cfg["eta"])
    return out
