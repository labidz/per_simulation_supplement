"""Numerical checks of the analytic results quoted in the manuscript."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from persist import core, experiments as ex

cfg = dict(ex.DEFAULT_CFG); cfg.update(json.load(open("configs/confirmatory.json"))["model"])
lam = cfg["lam"]; lamp = (1 + lam) / 2; ln_inv = np.log(1 / lamp)
TR = np.load("results/confirmatory_trajectories.npz")
out = {}
# Prop 1: closed-form persistence differences for R
pred = {"CRF-P75": np.log(1 / 0.75) / ln_inv, "P75-P50": np.log(0.75 / 0.5) / ln_inv, "P50-P25": np.log(2) / ln_inv}
print("Prop 1 closed-form differences (trials):", {k: round(v, 2) for k, v in pred.items()})
out["prop1"] = pred
# Corollary: time-shift of mean decision-variable trajectories (CRF vs P50)
ks = np.arange(0, 301)
def shift(model):
    a, b = TR[f"{model}|CRF_EXT|U"], TR[f"{model}|P50_EXT|U"]
    best = None
    for d in np.arange(0, 60, 0.1):
        k = np.arange(35, 150)
        err = np.mean((a[k] - np.interp(k - d, ks, b)) ** 2)
        if best is None or err < best[1]:
            best = (d, err)
    return best
sh = {m: shift(m) for m in ("R", "U", "IG", "Sp", "LPa")}
print("best-fit time shift of CRF vs P50 mean U (predicted %.1f):" % pred["P50-P25"], {m: (round(v[0], 1), float("%.2g" % v[1])) for m, v in sh.items()})
out["shift"] = {m: v[0] for m, v in sh.items()}
# Prop 4: information gain of the leaky learner vs no forgetting
res = {}
for lam_ig, N in ((0.95, 120), (1.0, 120), (1.0, 240)):
    c2 = dict(cfg, lam=lam_ig, N_main=N)
    cells = ex.build_cells(c2)
    for nm in ("CRF_CONT", "P50_CONT", "CUE_CONT"):
        sim = ex.simulate_cell(cells[nm], c2, 300, 424242, with_boc=False)
        res[f"lam={lam_ig},N={N},{nm[:3]}"] = float(sim["leaky"]["ig"][:, 0].mean())
ig0 = float(core.bald_bernoulli(0.5, 0.5))
print("prior information gain (nats): %.3f; leaky floor 1/(2(n_eff+1)) = %.4f nats = %.3f of prior" % (ig0, 1 / (2 * 22), (1 / 44) / ig0))
print("normalized IG at end of training:", {k: round(v, 3) for k, v in res.items()})
out["ig"] = res
# Prop 2 / 3 closed forms
h, eps = cfg["h"], cfg["cue_eps"]
kd = lambda p: np.log((1 - h) / h) / np.log(1 / (1 - p))
print("Prop 2: k_d(.5)=%.2f k_d(.75)=%.2f cued=%.2f" % (kd(0.5), kd(0.75), 2 * np.log((1 - h) / h) / np.log(1 / eps)))
json.dump(out, open("results/analytic_checks.json", "w"), indent=1, default=float)
