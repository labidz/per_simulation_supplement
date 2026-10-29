"""Confirmatory run. Reads the frozen config, calibrates on calibration seeds,
evaluates every cell on fresh evaluation seeds, checks the pre-specified
hypotheses and writes all outputs to results/."""
import csv, hashlib, json, sys, time
import numpy as np
sys.path.insert(0, ".")
from persist import core, experiments as ex, analysis

CFG_PATH = sys.argv[1] if len(sys.argv) > 1 else "configs/confirmatory.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/confirmatory"
raw = open(CFG_PATH, "rb").read()
spec = json.loads(raw)
cfg = dict(ex.DEFAULT_CFG); cfg.update(spec["model"]); cfg.update({k: spec["seeds"][k] for k in ("cal_seeds", "cal_seed", "eval_seeds", "eval_seed")})
print("config sha256", hashlib.sha256(raw).hexdigest(), flush=True)
cells = ex.build_cells(cfg)
t0 = time.time()
params = ex.calibrate_models(cfg, cells)
print("calibrated in %.1fs; shared cost c=%.4f" % (time.time() - t0, params["R"]["c"]), flush=True)
P = {m: {} for m in core.MODELS}
traj = {}
for name, cell in cells.items():
    sim = ex.simulate_cell(cell, cfg, cfg["eval_seeds"], cfg["eval_seed"])
    for m in core.MODELS:
        P[m][name] = ex.persistence_for(m, params, sim, cfg)
        if m != "N":
            V, S = core.model_VS(m, sim["leaky"], sim["boc"], pay=sim["pay"])
            U = V + params[m]["kappa"] * S
            traj[f"{m}|{name}|U"] = U.mean(0); traj[f"{m}|{name}|V"] = V.mean(0); traj[f"{m}|{name}|S"] = S.mean(0)
    print("cell %-12s done (%.0fs)" % (name, time.time() - t0), flush=True)
np.savez_compressed(OUT + "_persistence.npz", **{f"{m}|{c}": P[m][c].astype(np.int16) for m in P for c in P[m]})
np.savez_compressed(OUT + "_trajectories.npz", **traj)
json.dump(dict(cfg=cfg, params=params, config_sha256=hashlib.sha256(raw).hexdigest()),
          open(OUT + "_params.json", "w"), indent=1, default=float)

def dump(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

nb, sd = spec["seeds"]["bootstrap"], spec["seeds"]["bootstrap_seed"]
hyp = analysis.evaluate_hypotheses(P, spec["hypotheses"], nb, sd); dump(hyp, OUT + "_hypotheses.csv")
sig = analysis.signature_matrix(P, spec["signature_contrasts"], n_boot=nb, seed=sd); dump(sig, OUT + "_signature.csv")
pl = analysis.plausibility(P, cfg["T"]); dump(pl, OUT + "_plausibility.csv")
print("\nPRE-SPECIFIED HYPOTHESES")
for r in hyp:
    print("%-4s %-4s %-12s vs %-14s expect=%-5s diff=%8.2f [%8.2f, %8.2f] dz=%5.2f -> %s" % (
        r["id"], r["model"], r["a"], r["b"], r["expect"], r["diff"], r["lo"], r["hi"], r["dz"],
        "SUPPORTED" if r["supported"] else "NOT SUPPORTED"))
print("\nPLAUSIBILITY (mean persistence in continued-reward cells; T=%d)" % cfg["T"])
for r in pl:
    print(r)
print("total %.0fs" % (time.time() - t0))
