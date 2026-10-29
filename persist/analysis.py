"""Hypothesis evaluation and signature matrices from persistence arrays."""
from __future__ import annotations

import numpy as np

from . import core, experiments as ex


def _get(P, default_model, spec):
    if ":" in spec:
        m, c = spec.split(":")
        return P[m][c]
    return P[default_model][spec]


def evaluate_hypotheses(P, hyps, n_boot=4000, seed=7):
    """Check each pre-specified directional / equivalence hypothesis."""
    rows = []
    for h in hyps:
        qa = _get(P, h["model"], h["a"])
        qb = _get(P, h["model"], h["b"])
        r = ex.paired_contrast(qa, qb, n_boot=n_boot, seed=seed)
        if h["expect"] == "gt":
            ok = r["lo"] > 0
        elif h["expect"] == "lt":
            ok = r["hi"] < 0
        elif h["expect"] == "equiv":
            ok = (r["lo"] > -h["margin"]) and (r["hi"] < h["margin"])
        else:
            raise ValueError(h["expect"])
        rows.append(dict(id=h["id"], model=h["model"], a=h["a"], b=h["b"], expect=h["expect"],
                         margin=h.get("margin", ""), diff=r["diff"], lo=r["lo"], hi=r["hi"],
                         dz=r["dz"], supported=bool(ok)))
    return rows


def signature_matrix(P, contrasts, models=core.MODELS, n_boot=4000, seed=7):
    rows = []
    for c in contrasts:
        for m in models:
            r = ex.paired_contrast(P[m][c["a"]], P[m][c["b"]], n_boot=n_boot, seed=seed)
            rows.append(dict(contrast=c["id"], model=m, diff=r["diff"], lo=r["lo"], hi=r["hi"],
                             dz=r["dz"], p_sup=r["p_sup"], sign=ex.sign_label(r)))
    return rows


def plausibility(P, T, cont_cells=("CRF_CONT", "P50_CONT", "CUE_CONT"), frac=0.9, models=core.MODELS):
    rows = []
    for m in models:
        means = {c: float(P[m][c].mean()) for c in cont_cells}
        rows.append(dict(model=m, **means, plausible=all(v >= frac * T for v in means.values())))
    return rows
