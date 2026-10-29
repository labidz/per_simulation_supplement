"""Manuscript figures. Every figure answers one research question."""
import csv, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from persist import core, experiments as ex

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 200, "font.family": "DejaVu Sans"})
COL = {"N": "#999999", "R": "#000000", "CD": "#0072B2", "U": "#D55E00", "Sp": "#E69F00",
       "IG": "#009E73", "LPs": "#CC79A7", "LPa": "#8E3B8A", "H": "#56B4E9"}
LAB = {"N": "N null", "R": "R reward-only", "CD": "CD change detection", "U": "U uncertainty",
       "Sp": "Sp surprise", "IG": "IG information gain", "LPs": "LPs signed LP", "LPa": "LPa absolute LP",
       "H": "H CD + |LP|"}
spec = json.load(open("configs/confirmatory.json"))
cfg = dict(ex.DEFAULT_CFG); cfg.update(spec["model"])
P = np.load("results/confirmatory_persistence.npz")
TR = np.load("results/confirmatory_trajectories.npz")
prm = json.load(open("results/confirmatory_params.json"))["params"]
c_task = prm["R"]["c"]
T = cfg["T"]


def surv(q, T=T):
    ks = np.arange(0, T + 1)
    return ks, np.array([(q > k).mean() for k in ks])


# ---------------------------------------------------------------- Figure 1
fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
for ax, (cell, title) in zip(axes, [("CRF_EXT", "Extinction after 100% reward"), ("P50_EXT", "Extinction after 50% random reward"),
                                    ("DEV_P50", "Unpaid, unlearnable noise"), ("DEV_CUE_N20", "Unpaid, learning in progress")]):
    for m in ["R", "CD", "U", "Sp", "IG", "LPa", "H"]:
        U = TR[f"{m}|{cell}|U"][:80]
        ax.plot(U, color=COL[m], lw=1.3, label=LAB[m])
    ax.axhline(c_task, color="k", ls=":", lw=1, label="shared cost $c$")
    ax.set_title(title, fontsize=8.5); ax.set_xlabel("test trial k")
    ax.set_ylim(0, 1.35 if cell in ("CRF_EXT", "P50_EXT") else 0.45)
axes[0].set_ylabel("decision variable $U_k$ (mean over seeds)")
axes[0].annotate("LPa peaks at 4.6 (off scale)", xy=(10, 1.3), xytext=(22, 1.28), fontsize=6.5, color=COL["LPa"],
                 arrowprops=dict(arrowstyle="->", color=COL["LPa"], lw=0.8))
axes[0].legend(fontsize=6.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.92))
fig.tight_layout(); fig.savefig("figures/fig1_decision_variables.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 2
fig, axes = plt.subplots(2, 4, figsize=(11, 4.6), sharey=True)
sched = [("CRF_EXT", "100%", "#000000"), ("P75_EXT", "75%", "#0072B2"), ("P50_EXT", "50%", "#E69F00"), ("P25_EXT", "25%", "#D55E00")]
for ax, m in zip(axes.ravel(), ["R", "CD", "U", "Sp", "IG", "LPs", "LPa", "H"]):
    for cell, lab, col in sched:
        ks, s = surv(P[f"{m}|{cell}"])
        ax.step(ks, s, color=col, lw=1.3, label=lab, where="post")
    ax.set_title(LAB[m], fontsize=8.5); ax.set_xlim(0, {"CD": 30, "H": 40, "LPs": 100}.get(m, 200))
for ax in axes[1]:
    ax.set_xlabel("test trial")
for ax in axes[0]:
    ax.set_xlabel("test trial")
for ax in axes[:, 0]:
    ax.set_ylabel("proportion still engaged")
axes[0, 0].legend(title="training reward", fontsize=7, title_fontsize=7, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig("figures/fig2_extinction_survival.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 3
rows = list(csv.DictReader(open("results/confirmatory_signature.csv")))
cons = []
for r in rows:
    if r["contrast"] not in cons:
        cons.append(r["contrast"])
models = list(core.MODELS)
Z = np.zeros((len(cons), len(models))); D = np.zeros_like(Z); SG = [[None] * len(models) for _ in cons]
for r in rows:
    i, j = cons.index(r["contrast"]), models.index(r["model"])
    Z[i, j], D[i, j], SG[i][j] = float(r["dz"]), float(r["diff"]), r["sign"]
fig, ax = plt.subplots(figsize=(8.6, 4.2))
im = ax.imshow(np.clip(Z, -3, 3), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
ax.set_xticks(range(len(models))); ax.set_xticklabels(models); ax.set_yticks(range(len(cons)))
ax.set_yticklabels(["PREE: 50% − 100%", "DOSE: 25% − 75%", "LEARN: noise − cued (extinction)", "REVERSAL − extinction", "LEAN − none (50% training)",
                    "UNPAID: noise − learning", "UNPAID: mastered − learning", "UNPAID: relearning − mastered"])
for i in range(len(cons)):
    for j in range(len(models)):
        if models[j] == "N":
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#888888"); continue
        val = 0.0 if abs(D[i, j]) < 0.05 else D[i, j]
        ax.text(j, i, f"{val:.0f}" if abs(val) >= 10 else f"{val:.1f}", ha="center", va="center",
                fontsize=7.2, color="white" if abs(Z[i, j]) > 1.8 else "black",
                fontweight="bold" if SG[i][j] in "+-" else "normal")
cb = fig.colorbar(im, ax=ax, fraction=0.03); cb.set_label("standardized paired effect $d_z$ (clipped at ±3)")
ax.set_title("Signature matrix: mean difference in persistence (trials); bold = 95% CI excludes 0", fontsize=8.5)
fig.tight_layout(); fig.savefig("figures/fig3_signature_matrix.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 4 (unpaid persistence vs intrinsic weight)
cells = ex.build_cells(cfg)
NAMES = ["DEV_P50", "DEV_CUE_N20", "DEV_CUE", "DEV_CUEREV"]
sims = {n: ex.simulate_cell(cells[n], cfg, 800, 888000) for n in NAMES}
cal = ex.simulate_cell(cells[cfg["ref_cell"]], cfg, 1500, 111)
rhos = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
res = {m: {n: [] for n in NAMES} for m in ["U", "Sp", "IG", "LPs", "LPa", "H"]}
for rho in rhos:
    pr = ex.calibrate_models(dict(cfg, rho=rho), cells, models=("R", "U", "Sp", "IG", "LPs", "LPa", "H"), sim=cal)
    for m in res:
        for n in NAMES:
            res[m][n].append(ex.persistence_for(m, pr, sims[n], cfg).mean())
json.dump({"rhos": rhos, "persistence": res}, open("results/unpaid_vs_rho.json", "w"))
fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
for ax, (n, t) in zip(axes, [("DEV_P50", "Unlearnable noise"), ("DEV_CUE_N20", "Learning in progress"),
                             ("DEV_CUE", "Mastered"), ("DEV_CUEREV", "Relearning after reversal")]):
    for m in res:
        ax.plot(rhos, res[m][n], marker="o", ms=3, color=COL[m], lw=1.2, label=LAB[m])
    ax.set_xscale("log", base=2); ax.set_title(t, fontsize=8.5); ax.set_xlabel("intrinsic weight $\\rho$")
    ax.axvline(0.5, color="#bbbbbb", lw=0.8)
axes[0].set_ylabel("mean unpaid persistence (trials)")
axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig4_unpaid_vs_weight.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 5 (identifiability)
import os
rec = json.load(open("results/recovery_summary.json"))
flex = json.load(open("results/recovery_flex_summary.json")) if os.path.exists("results/recovery_flex_summary.json") else None
mods = rec["models"]
C = np.array(rec["conf_n40"])
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), gridspec_kw={"width_ratios": [1, 1.05, 1.15]})
ax = axes[0]
im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(mods))); ax.set_xticklabels(mods, fontsize=7); ax.set_yticks(range(len(mods))); ax.set_yticklabels(mods, fontsize=7)
ax.set_xlabel("winning model (group AIC, n = 40)"); ax.set_ylabel("generating model")
for i in range(len(mods)):
    for j in range(len(mods)):
        if C[i, j] >= 0.02:
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6.3, color="white" if C[i, j] > 0.55 else "black")
ax.set_title("A  Model recovery (matched hyperparameters)", fontsize=8.5, loc="left")
ax = axes[1]
ns = sorted({int(k.split("|")[1]) for k in rec["power"]})
for m in mods:
    ax.plot(ns, [rec["power"].get(f"{m}|{n}", np.nan) for n in ns], marker="o", ms=3, color=COL[m], lw=1.2, label=LAB[m])
ax.axhline(0.9, color="k", ls=":", lw=1); ax.set_xscale("log", base=2)
ax.set_xlabel("participants (each completes all five blocks)"); ax.set_ylabel("P(correct group-level identification)")
ax.set_title("B  Power to identify the generating model", fontsize=8.5, loc="left"); ax.set_ylim(0, 1.02)
ax.legend(fontsize=5.8, frameon=False, ncol=2, loc="lower right")
ax = axes[2]
x = np.arange(len(mods)); wd = 0.2
series = [("matched", np.diag(np.array(rec["conf_n40"])), "#2b6cb0"),
          ("mismatch (on-grid), fixed", np.diag(np.array(rec["conf_n40_mismatch"])), "#d9534f")]
if flex:
    series += [("mismatch (off-grid), fixed", np.diag(np.array(flex["fixed_n40"])), "#f0ad4e"),
               ("mismatch (off-grid), hyperparameters free", np.diag(np.array(flex["flex_n40"])), "#5cb85c")]
for i, (lab, vals, col) in enumerate(series):
    ax.bar(x + (i - (len(series) - 1) / 2) * wd, vals, wd, label=lab, color=col)
ax.set_xticks(x); ax.set_xticklabels(mods, fontsize=7); ax.set_ylim(0, 1.05)
ax.set_ylabel("P(correct), n = 40"); ax.set_title("C  Stress test: misspecified internal model", fontsize=8.5, loc="left")
ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
fig.tight_layout(); fig.savefig("figures/fig5_identifiability.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Supplementary figure: robustness of the hypotheses
H = list(csv.DictReader(open("results/robustness_hypotheses.csv")))
variants = []
for r in H:
    if r["variant"] not in variants:
        variants.append(r["variant"])
hids = [h["id"] for h in spec["hypotheses"]]
M = np.zeros((len(hids), len(variants)))
for r in H:
    M[hids.index(r["id"]), variants.index(r["variant"])] = r["supported"] == "True"
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.imshow(M, cmap="Greens", vmin=0, vmax=1.4, aspect="auto")
ax.set_xticks(range(len(variants))); ax.set_xticklabels(variants, rotation=60, ha="right", fontsize=7)
ax.set_yticks(range(len(hids))); ax.set_yticklabels(hids)
for i in range(len(hids)):
    for j in range(len(variants)):
        ax.text(j, i, "✓" if M[i, j] else "✗", ha="center", va="center", fontsize=8, color="white" if M[i, j] else "#b00020")
ax.set_title("Pre-specified hypotheses under one-factor-at-a-time changes (✓ supported, ✗ not supported)", fontsize=8.5)
fig.tight_layout(); fig.savefig("figures/figS1_robustness.png"); plt.close(fig)
print("figures written")
