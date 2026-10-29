"""Pre-specified robustness analysis (one factor at a time, fresh seeds).
Every variant recalibrates the shared cost and intrinsic weights on its own
calibration seeds, then re-evaluates all pre-specified hypotheses and the
signature contrasts."""
import csv, json, sys, time, copy
import numpy as np
sys.path.insert(0, ".")
from persist import core, experiments as ex, analysis

spec = json.load(open("configs/confirmatory.json"))
base = dict(ex.DEFAULT_CFG); base.update(spec["model"])
S_EVAL, S_CAL, SEED_EVAL, SEED_CAL = 600, 1000, 777000, 333
NEED = ["CRF_CONT", "P50_CONT", "CUE_CONT", "CRF_EXT", "P75_EXT", "P50_EXT", "P25_EXT", "CUE_EXT",
        "P50_LEAN", "CUE_REV", "DEV_P50", "DEV_CUE", "DEV_CUE_N20", "DEV_CUEREV"]
MODELS = tuple(core.MODELS) + ("LPp",)
t0 = time.time()

def simulate(cfg, names):
    cells = ex.build_cells(cfg)
    return {n: ex.simulate_cell(cells[n], cfg, S_EVAL, SEED_EVAL) for n in names}, cells

def calsim(cfg, cells):
    return ex.simulate_cell(cells[cfg["ref_cell"]], cfg, S_CAL, SEED_CAL)

def run_variant(label, cfg, sims, cells, rows_h, rows_s, rows_p, cal=None):
    cal = cal if cal is not None else calsim(cfg, cells)
    params = ex.calibrate_models(cfg, cells, models=MODELS, sim=cal)
    P = {m: {n: ex.persistence_for(m, params, sims[n], cfg) for n in sims} for m in MODELS}
    for r in analysis.evaluate_hypotheses(P, spec["hypotheses"], 1500, 7):
        rows_h.append(dict(variant=label, **r))
    contrasts = [c for c in spec["signature_contrasts"] if c["a"] in sims and c["b"] in sims]
    for r in analysis.signature_matrix(P, contrasts, models=MODELS, n_boot=1500, seed=7):
        rows_s.append(dict(variant=label, **r))
    for r in analysis.plausibility(P, cfg["T"], models=MODELS):
        rows_p.append(dict(variant=label, **r))
    print("%-22s done %.0fs  c=%.3f" % (label, time.time() - t0, params["R"]["c"]), flush=True)

H, SG, PL = [], [], []
sims0, cells0 = simulate(base, NEED)
cal0 = calsim(base, cells0)
run_variant("nominal", base, sims0, cells0, H, SG, PL, cal0)

# --- rho (intrinsic weight): same streams, new kappa
for rho in (0.125, 0.25, 1.0, 2.0):
    cfg = dict(base, rho=rho)
    run_variant(f"rho={rho}", cfg, sims0, cells0, H, SG, PL, cal0)
# --- LP window
for w in (5, 20, 40):
    cfg = dict(base, w=w)
    sims = {n: ex.set_window(s, cfg, w) for n, s in sims0.items()}
    cal = ex.set_window(cal0, cfg, w)
    run_variant(f"w={w}", cfg, sims, cells0, H, SG, PL, cal)
# --- decision noise and cost target (recalibrate c)
for tau in (0.005, 0.03):
    cfg = dict(base, tau=tau); run_variant(f"tau={tau}", cfg, sims0, cells0, H, SG, PL, cal0)
for tg in (20.0, 80.0):
    cfg = dict(base, target=tg); run_variant(f"target={tg:g}", cfg, sims0, cells0, H, SG, PL, cal0)
# --- world-model leak (re-simulate leaky signals only)
for lam in (0.90, 0.98):
    cfg = dict(base, lam=lam)
    sims = {n: ex.resimulate_leaky(s, cfg) for n, s in sims0.items()}
    run_variant(f"lam={lam}", cfg, sims, cells0, H, SG, PL, ex.resimulate_leaky(cal0, cfg))
# --- change-point hazard (re-simulate everything)
for h in (0.005, 0.05):
    cfg = dict(base, h=h); sims, cells = simulate(cfg, NEED)
    run_variant(f"h={h}", cfg, sims, cells, H, SG, PL)
# --- cue noise / learnability strength
for eps in (0.0, 0.1, 0.2):
    cfg = dict(base, cue_eps=eps); sims, cells = simulate(cfg, NEED)
    run_variant(f"eps={eps}", cfg, sims, cells, H, SG, PL)

def dump(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
dump(H, "results/robustness_hypotheses.csv"); dump(SG, "results/robustness_signature.csv"); dump(PL, "results/robustness_plausibility.csv")
print("robustness complete %.0fs" % (time.time() - t0))
