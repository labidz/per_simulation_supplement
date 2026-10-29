"""EXPLORATORY pilot (seeds never used in the confirmatory run)."""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from persist import core, experiments as ex

cfg = dict(ex.DEFAULT_CFG)
cfg.update(cal_seeds=600, cal_seed=9001, eval_seeds=250, eval_seed=9500)
cells = ex.build_cells(cfg)
t0 = time.time()
sims = {n: ex.simulate_cell(cells[n], cfg, cfg["eval_seeds"], cfg["eval_seed"]) for n in cells}
print("simulated cells in", round(time.time() - t0, 1), "s")
for rho in (0.25, 0.5, 1.0, 2.0):
    cfg["rho"] = rho
    params = ex.calibrate_models(cfg, cells)
    print("\nrho =", rho, " c =", round(params["R"]["c"], 4), " kappas:",
          {m: round(params[m]["kappa"], 2) for m in core.INTRINSIC_MODELS})
    print("%-12s" % "cell", " ".join("%6s" % m for m in core.MODELS))
    for n in cells:
        print("%-12s" % n, " ".join("%6.1f" % ex.persistence_for(m, params, sims[n], cfg).mean() for m in core.MODELS))
