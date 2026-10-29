"""Parameter recovery, model recovery and power analysis for the proposed
five-block within-subject design (fresh seeds). Includes a hyperparameter
misspecification stress test."""
import csv, json, sys, time
import numpy as np
sys.path.insert(0, ".")
from persist import core, experiments as ex, recovery as rc

spec = json.load(open("configs/confirmatory.json"))
cfg = dict(ex.DEFAULT_CFG); cfg.update(spec["model"])
params = json.load(open("results/confirmatory_params.json"))["params"]
units = {m: params[m]["kappa"] / cfg["rho"] for m in core.INTRINSIC_MODELS}
T, NP = cfg["T"], 300
MODELS = core.MODELS
rng = np.random.default_rng(2026)
t0 = time.time()
pool = rc.build_pool(cfg, NP, seed=555000)
idx = np.tile(np.arange(NP)[:, None], (1, len(rc.BLOCKS)))
print("pool built %.0fs" % (time.time() - t0), flush=True)

ll_by_true, par_rec, parts_all, Q_all = {}, {}, {}, {}
for true in MODELS:
    parts = rc.sample_participants(true, NP, rng)
    Q = rc.generate_quit_times(true, parts, pool, idx, units, T, rng)
    ll, best = rc.fit_all(pool, idx, Q, units, T)
    ll_by_true[true] = ll
    par_rec[true] = rc.parameter_recovery(true, parts, best[true])
    parts_all[true], Q_all[true] = parts, Q
    print("fitted data from %-3s  %.0fs  mean quit times by block %s" % (true, time.time() - t0, np.round(Q.mean(0), 1)), flush=True)
np.savez_compressed("results/recovery_ll.npz", **{f"ll|{m}": ll_by_true[m] for m in MODELS},
                    **{f"Q|{m}": Q_all[m] for m in MODELS})

k = np.array([rc.NPAR[m] for m in MODELS], float)
def indiv_conf(ll_by_true):
    C = np.zeros((len(MODELS), len(MODELS)))
    for t, m in enumerate(MODELS):
        aic = -2 * ll_by_true[m] + 2 * k[None, :]
        w = aic.argmin(1)
        for j in range(len(MODELS)):
            C[t, j] = (w == j).mean()
    return C
C_ind = indiv_conf(ll_by_true)
ns = [5, 10, 20, 40, 80, 160]
power = rc.power_curve(ll_by_true, ns, n_rep=300, seed=11)
C20 = rc.confusion_matrix(ll_by_true, 20, 300, seed=12); C40 = rc.confusion_matrix(ll_by_true, 40, 300, seed=13)
C40_bic = rc.confusion_matrix(ll_by_true, 40, 300, seed=14, criterion="bic")

# --- misspecification: participants generated with different hyperparameters
cfg_gen = dict(cfg, lam=0.90, h=0.05, w=5, eta=0.2)
pool_gen = rc.build_pool(cfg_gen, NP, seed=555000)       # identical streams, different internal model
NM = 150
ll_mis = {}
for true in MODELS:
    parts = rc.sample_participants(true, NM, rng)
    Q = rc.generate_quit_times(true, parts, pool_gen, idx[:NM], units, T, rng)
    ll, _ = rc.fit_all(pool, idx[:NM], Q, units, T)       # fitted with nominal internal model
    ll_mis[true] = ll
    print("mismatch data from %-3s  %.0fs" % (true, time.time() - t0), flush=True)
C_ind_mis = indiv_conf(ll_mis)
power_mis = rc.power_curve(ll_mis, [10, 20, 40, 80], n_rep=300, seed=15)
C40_mis = rc.confusion_matrix(ll_mis, 40, 300, seed=16)

out = dict(models=list(MODELS), blocks=rc.BLOCKS, n_per_model=NP, n_mismatch=NM,
           param_recovery=par_rec, indiv_confusion=C_ind.tolist(),
           conf_n20=C20.tolist(), conf_n40=C40.tolist(), conf_n40_bic=C40_bic.tolist(),
           power={f"{a}|{b}": v for (a, b), v in power.items()},
           indiv_confusion_mismatch=C_ind_mis.tolist(), conf_n40_mismatch=C40_mis.tolist(),
           power_mismatch={f"{a}|{b}": v for (a, b), v in power_mis.items()})
json.dump(out, open("results/recovery_summary.json", "w"), indent=1)
print("\nindividual-level accuracy (AIC):", {m: round(C_ind[i, i], 2) for i, m in enumerate(MODELS)})
print("group n=40 accuracy:", {m: round(C40[i, i], 2) for i, m in enumerate(MODELS)})
print("group n=40 accuracy under mismatch:", {m: round(C40_mis[i, i], 2) for i, m in enumerate(MODELS)})
print("parameter recovery (Spearman):", par_rec)
print("done %.0fs" % (time.time() - t0))
