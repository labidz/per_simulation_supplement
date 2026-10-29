import numpy as np
from scipy.special import expit
from persist import core, experiments as ex, recovery as rc


def small_cfg():
    cfg = dict(ex.DEFAULT_CFG)
    cfg.update(T=40, N_main=40, N_short=10, target=10.0)
    return cfg


def test_generation_matches_direct_quit_rule():
    cfg = small_cfg()
    pool = rc.build_pool(cfg, 25, seed=3)
    rng = np.random.default_rng(0)
    parts = rc.sample_participants("R", 25, rng)
    idx = np.tile(np.arange(25)[:, None], (1, len(rc.BLOCKS)))
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    Q = rc.generate_quit_times("R", parts, pool, idx, {}, cfg["T"], rng_a)
    # manual recomputation with an identical random stream
    for j, b in enumerate(rc.BLOCKS):
        v = rng_b.random((25, cfg["T"]))
        V, _ = pool[b]["R"]
        for i in range(25):
            p = expit((V[i, :cfg["T"]] - parts["c"][i]) / parts["tau"][i])
            dis = ~(v[i] < p)
            q = dis.argmax() if dis.any() else cfg["T"]
            assert Q[i, j] == q


def test_fit_loglik_equals_manual_grid_maximum():
    cfg = small_cfg()
    pool = rc.build_pool(cfg, 6, seed=4)
    rng = np.random.default_rng(1)
    parts = rc.sample_participants("R", 6, rng)
    idx = np.tile(np.arange(6)[:, None], (1, len(rc.BLOCKS)))
    Q = rc.generate_quit_times("R", parts, pool, idx, {}, cfg["T"], rng)
    ll, best = rc.fit_model("R", pool, idx, Q, {}, cfg["T"])
    for i in range(6):
        manual = []
        for c in rc.GRID["c"]:
            for tau in rc.GRID["tau"]:
                tot = 0.0
                for j, b in enumerate(rc.BLOCKS):
                    V, _ = pool[b]["R"]
                    z = (V[i, :cfg["T"]] - c) / tau
                    q = Q[i, j]
                    tot += np.sum(np.log(expit(z[:q]))) if q > 0 else 0.0
                    if q < cfg["T"]:
                        tot += np.log(1.0 - expit(z[q]))
                manual.append(tot)
        assert np.isclose(ll[i], max(manual), atol=1e-6)


def test_null_model_fit_recovers_hazard_and_wins_on_null_data():
    cfg = small_cfg()
    pool = rc.build_pool(cfg, 80, seed=6)
    rng = np.random.default_rng(2)
    n = 80
    parts = dict(q=np.full(n, 0.03))
    idx = np.tile(np.arange(n)[:, None], (1, len(rc.BLOCKS)))
    Q = rc.generate_quit_times("N", parts, pool, idx, {}, cfg["T"], rng)
    ll, best = rc.fit_model("N", pool, idx, Q, {}, cfg["T"])
    assert 0.015 < np.median(best["q"]) < 0.06
    units = {m: 1.0 for m in core.INTRINSIC_MODELS}
    llall, _ = rc.fit_all(pool, idx, Q, units, cfg["T"], models=["N", "R"])
    assert rc.group_winner(llall, np.arange(n), ["N", "R"]) == 0


def test_group_winner_penalises_parameters():
    ll = np.zeros((10, 3))          # identical fit quality: simplest model must win
    assert rc.group_winner(ll, np.arange(10), ["N", "R", "U"]) == 0
