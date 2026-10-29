"""Does fitting the internal-model hyperparameters restore model recovery under
misspecification? Participants are generated with OFF-GRID hyperparameters
(lam=0.92, eta=0.15, w=7, h=0.035); fits either fix the nominal values or take
the best of a small hyperparameter grid (with an AIC penalty per free hyperparameter)."""
import json, sys, time, itertools
import numpy as np
sys.path.insert(0, ".")
from persist import core, experiments as ex, recovery as rc

spec = json.load(open("configs/confirmatory.json"))
cfg = dict(ex.DEFAULT_CFG); cfg.update(spec["model"])
params = json.load(open("results/confirmatory_params.json"))["params"]
units = {m: params[m]["kappa"] / cfg["rho"] for m in core.INTRINSIC_MODELS}
T, NP, NM = cfg["T"], 300, 150
MODELS = list(core.MODELS)
LAM, ETA, WIN, HAZ = [0.90, 0.95, 0.98], [0.1, 0.2], [5, 10, 20], [0.02, 0.05]
t0 = time.time()
cells = ex.build_cells(cfg)

# ---- raw signals for every hyperparameter setting, per block (same streams everywhere)
raw = {}
for b in rc.BLOCKS:
    sim = ex.simulate_cell(cells[b], cfg, NP, 555000, with_boc=False)
    cue, o, na, pay = sim["cue"], sim["o"], sim["n_acq"], sim["pay"]
    leaky = {(l, e): core.leaky_signals(cue, o, na, T, lam=l, eta=e, w=10) for l in LAM for e in ETA}
    boc = {h: core.bocpd_signals(cue, o, na, T, h=h, w=10) for h in HAZ}
    raw[b] = (leaky, boc, na, pay)
print("raw signals %.0fs" % (time.time() - t0), flush=True)

def make_pool(model, lam=0.95, eta=0.1, w=10, h=0.02):
    pool = {}
    for b in rc.BLOCKS:
        leaky, boc, na, pay = raw[b]
        lk, bc = leaky[(lam, eta)], boc[h]
        ov = (core.lp_traj(lk["loss"], na, T, w), core.lp_traj(bc["loss"], na, T, w))
        pool[b] = {model: core.model_VS(model, lk, bc, pay=pay, lp_override=ov)}
    return pool

COMBOS = {"R": [dict(lam=l) for l in LAM], "CD": [dict(h=h) for h in HAZ], "U": [dict(lam=l) for l in LAM],
          "Sp": [dict(lam=l, eta=e) for l in LAM for e in ETA], "IG": [dict(lam=l) for l in LAM],
          "LPs": [dict(lam=l, w=w) for l in LAM for w in WIN], "LPa": [dict(lam=l, w=w) for l in LAM for w in WIN],
          "H": [dict(h=h, w=w) for h in HAZ for w in WIN]}
DOF = {"N": 0, "R": 1, "CD": 1, "U": 1, "Sp": 2, "IG": 1, "LPs": 2, "LPa": 2, "H": 2}
pools = {m: [make_pool(m, **kw) for kw in COMBOS[m]] for m in COMBOS}
nominal = rc.build_pool(cfg, NP, seed=555000)
print("pools built %.0fs" % (time.time() - t0), flush=True)

cfg_gen = dict(cfg, lam=0.92, eta=0.15, w=7, h=0.035)
pool_gen = rc.build_pool(cfg_gen, NP, seed=555000)
idx = np.tile(np.arange(NP)[:, None], (1, len(rc.BLOCKS)))[:NM]
import os
CK = "results/recovery_flex_ll.npz"
ll_fix, ll_flex = {}, {}
if os.path.exists(CK):
    z = np.load(CK)
    for m in MODELS:
        if f"fix|{m}" in z:
            ll_fix[m], ll_flex[m] = z[f"fix|{m}"], z[f"flex|{m}"]
for true in MODELS:
    if true in ll_fix:
        print("resumed", true, flush=True); continue
    rng = np.random.default_rng(404 + MODELS.index(true))
    parts = rc.sample_participants(true, NM, rng)
    Q = rc.generate_quit_times(true, parts, pool_gen, idx, units, T, rng)
    lf, _ = rc.fit_all(nominal, idx, Q, units, T)
    fx = np.zeros_like(lf)
    for i, m in enumerate(MODELS):
        if m == "N":
            fx[:, i] = lf[:, i]; continue
        best = np.full(NM, -np.inf)
        for pl in pools[m]:
            l, _ = rc.fit_model(m, pl, idx, Q, units, T)
            best = np.maximum(best, l)
        fx[:, i] = best
    ll_fix[true], ll_flex[true] = lf, fx
    np.savez(CK, **{f"fix|{m}": ll_fix[m] for m in ll_fix}, **{f"flex|{m}": ll_flex[m] for m in ll_flex})
    print("true %-3s done %.0fs" % (true, time.time() - t0), flush=True)
    if time.time() - t0 > float(os.environ.get("MAXSEC", "1e9")):
        print("time budget reached; rerun to resume", flush=True); sys.exit(0)

def winner(ll, rows, dof):
    k = np.array([rc.NPAR[m] + dof[m] for m in MODELS], float)
    return int((-2 * ll[rows].sum(0) + len(rows) * 2 * k).argmin())
def conf(llset, n, dof, reps=300, seed=0):
    r = np.random.default_rng(seed); C = np.zeros((9, 9))
    for t, m in enumerate(MODELS):
        for _ in range(reps):
            C[t, winner(llset[m], r.choice(NM, n, replace=False), dof)] += 1
    return C / reps
zero = {m: 0 for m in MODELS}
out = {"models": MODELS, "generating": dict(lam=0.92, eta=0.15, w=7, h=0.035)}
for n in (10, 20, 40, 80):
    out[f"fixed_n{n}"] = conf(ll_fix, n, zero, seed=n).tolist()
    out[f"flex_n{n}"] = conf(ll_flex, n, DOF, seed=n + 1).tolist()
json.dump(out, open("results/recovery_flex_summary.json", "w"), indent=1)
for n in (20, 40):
    print("n=%d fixed-hyperparameter accuracy:" % n, {m: round(out[f"fixed_n{n}"][i][i], 2) for i, m in enumerate(MODELS)})
    print("n=%d flexible-hyperparameter accuracy:" % n, {m: round(out[f"flex_n{n}"][i][i], 2) for i, m in enumerate(MODELS)})
print("done %.0fs" % (time.time() - t0))
