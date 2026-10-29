import numpy as np
from persist import core, experiments as ex, analysis


def test_hypothesis_decision_logic():
    rng = np.random.default_rng(0)
    base = rng.integers(20, 40, 400).astype(float)
    P = {"R": {"A": base + 10.0, "B": base, "C": base + 0.2 * rng.standard_normal(400)}}
    hyps = [dict(id="gt_ok", model="R", a="A", b="B", expect="gt"),
            dict(id="gt_bad", model="R", a="B", b="A", expect="gt"),
            dict(id="lt_ok", model="R", a="B", b="A", expect="lt"),
            dict(id="equiv_ok", model="R", a="C", b="B", expect="equiv", margin=1.0),
            dict(id="equiv_bad", model="R", a="A", b="B", expect="equiv", margin=1.0)]
    out = {r["id"]: r["supported"] for r in analysis.evaluate_hypotheses(P, hyps, n_boot=500, seed=1)}
    assert out == {"gt_ok": True, "gt_bad": False, "lt_ok": True, "equiv_ok": True, "equiv_bad": False}


def test_set_window_changes_only_learning_progress():
    cfg = dict(ex.DEFAULT_CFG, T=40, N_main=40, N_short=10)
    cell = ex.build_cells(cfg)["CUE_REV"]
    sim = ex.simulate_cell(cell, cfg, 30, 5)
    s2 = ex.set_window(sim, cfg, 5)
    assert np.array_equal(s2["leaky"]["V"], sim["leaky"]["V"]) and np.array_equal(s2["boc"]["V"], sim["boc"]["V"])
    assert not np.allclose(s2["leaky"]["lp"], sim["leaky"]["lp"])
    assert np.allclose(s2["leaky"]["lp"], core.lp_traj(sim["leaky"]["loss"], sim["n_acq"], cfg["T"], 5))


def test_resimulate_leaky_matches_fresh_simulation():
    cfg = dict(ex.DEFAULT_CFG, T=40, N_main=40, N_short=10)
    cells = ex.build_cells(cfg)
    sim = ex.simulate_cell(cells["P50_EXT"], cfg, 30, 6)
    cfg2 = dict(cfg, lam=0.9)
    fresh = ex.simulate_cell(cells["P50_EXT"], cfg2, 30, 6)
    redo = ex.resimulate_leaky(sim, cfg2)
    for k in ("V", "assoc", "ent", "ig", "lp"):
        assert np.allclose(fresh["leaky"][k], redo["leaky"][k])
    assert np.array_equal(sim["boc"]["V"], redo["boc"]["V"])
