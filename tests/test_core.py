import numpy as np
import pytest
from scipy import integrate, stats
from scipy.special import expit

from persist import core


# ---------------------------------------------------------------- streams
def test_phase_probs_values():
    assert core.phase_probs("crf") == (1.0, 1.0)
    assert core.phase_probs("random", p=0.25) == (0.25, 0.25)
    assert core.phase_probs("ext") == (0.0, 0.0)
    a, b = core.phase_probs("cued", eps=0.05)
    assert np.isclose(a, 0.95) and np.isclose(b, 0.05)
    # reversal swaps the cue-outcome mapping and preserves the marginal reward rate
    ra, rb = core.phase_probs("cued_rev", eps=0.05)
    assert (ra, rb) == (b, a)
    assert np.isclose((a + b) / 2, (ra + rb) / 2)
    with pytest.raises(ValueError):
        core.phase_probs("nonsense")


def test_noise_is_reproducible_and_seed_sensitive():
    c1, u1, v1 = core.draw_noise(50, 30, 20, seed=7)
    c2, u2, v2 = core.draw_noise(50, 30, 20, seed=7)
    c3, u3, v3 = core.draw_noise(50, 30, 20, seed=8)
    assert np.array_equal(c1, c2) and np.array_equal(u1, u2) and np.array_equal(v1, v2)
    assert not np.array_equal(u1, u3)
    assert c1.shape == (50, 50) and v1.shape == (50, 20)


def test_outcome_frequencies_match_schedule():
    cue, u, _ = core.draw_noise(4000, 100, 100, seed=1)
    o = core.make_outcomes(cue, u, 100, core.phase_probs("cued"), core.phase_probs("ext"))
    acq, tst = o[:, :100], o[:, 100:]
    c_acq = cue[:, :100]
    assert abs(acq[c_acq == 0].mean() - 0.95) < 0.01
    assert abs(acq[c_acq == 1].mean() - 0.05) < 0.01
    assert tst.sum() == 0  # pure extinction delivers no reward
    o_crf = core.make_outcomes(cue, u, 100, core.phase_probs("crf"), core.phase_probs("crf"))
    assert o_crf.min() == 1


def test_conditions_share_noise_so_outcomes_are_paired():
    cue, u, _ = core.draw_noise(500, 60, 40, seed=3)
    o_a = core.make_outcomes(cue, u, 60, core.phase_probs("random", p=0.5), core.phase_probs("ext"))
    o_b = core.make_outcomes(cue, u, 60, core.phase_probs("random", p=0.25), core.phase_probs("ext"))
    # p=0.25 rewards are a subset of p=0.5 rewards under common random numbers
    assert np.all(o_b <= o_a)


# ---------------------------------------------------------------- information gain
@pytest.mark.parametrize("a,b", [(0.5, 0.5), (2.0, 3.0), (10.5, 10.5), (19.5, 1.5)])
def test_bald_matches_numerical_integration(a, b):
    def h(p):
        return -(p * np.log(p) + (1 - p) * np.log1p(-p))
    m = a / (a + b)
    e_h = integrate.quad(lambda p: h(p) * stats.beta.pdf(p, a, b), 0, 1, limit=200)[0]
    numeric = h(m) - e_h
    assert np.isclose(core.bald_bernoulli(a, b), numeric, atol=1e-6)


def test_bald_nonnegative_and_decays_like_one_over_2n():
    for n in (20, 50, 200):
        val = float(core.bald_bernoulli(n / 2, n / 2))
        assert val > 0
        assert abs(val - 1.0 / (2 * (n + 1))) / val < 0.05
    # more counts -> less information at the same mean
    assert core.bald_bernoulli(5, 5) > core.bald_bernoulli(50, 50)
    # near-deterministic sources carry less information than fair noise at equal counts
    assert core.bald_bernoulli(19.5, 1.5) < core.bald_bernoulli(10.5, 10.5)


# ---------------------------------------------------------------- learning progress
def test_lp_signed_matches_analytic_step():
    w, n0, lo, hi = 10, 40, 0.2, 1.4
    loss = np.concatenate([np.full(n0, lo), np.full(60, hi)])[None, :]
    for j in range(0, w + 1):
        lp = core.lp_from_losses(loss, n0 + j, w)[0]
        assert np.isclose(lp, -(j / w) * (hi - lo))
    # after the step enters the older window, progress turns positive when loss falls again
    loss2 = np.concatenate([np.full(n0, lo), np.full(w, hi), np.full(2 * w, lo)])[None, :]
    assert core.lp_from_losses(loss2, n0 + w + w, w)[0] > 0


def test_lp_zero_before_two_windows():
    loss = np.random.default_rng(0).random((3, 15))
    assert np.all(core.lp_from_losses(loss, 19, 10) == 0)


# ---------------------------------------------------------------- leaky model
def test_leaky_mean_closed_form_in_extinction():
    lam, a0, b0, p = 0.95, 0.5, 0.5, 0.5
    n_acq, n_test, S = 300, 30, 6000
    cue, u, _ = core.draw_noise(S, n_acq, n_test, seed=11)
    o = core.make_outcomes(cue, u, n_acq, core.phase_probs("random", p=p), core.phase_probs("ext"))
    sig = core.leaky_signals(cue, o, n_acq, n_test, lam=lam, a0=a0, b0=b0)
    n_star = 1.0 / (1.0 - lam)
    for k in (0, 5, 15, 30):
        pred = (a0 + p * n_star * (0.5 + 0.5 * lam) ** k) / (a0 + b0 + n_star)
        assert abs(sig["V"][:, k].mean() - pred) < 0.01


def test_leaky_signals_bounded_and_shapes():
    cue, u, _ = core.draw_noise(100, 50, 30, seed=2)
    o = core.make_outcomes(cue, u, 50, core.phase_probs("cued"), core.phase_probs("ext"))
    sig = core.leaky_signals(cue, o, 50, 30)
    for key in ("V", "assoc", "ig"):
        assert sig[key].shape == (100, 31)
        assert sig[key].min() >= 0
    assert sig["V"].max() <= 1 and sig["assoc"].max() <= 1 and sig["ig"].max() <= 1.0 + 1e-9


def test_binary_entropy_and_uncertainty_signal():
    assert np.isclose(core.binary_entropy_bits(0.5), 1.0)
    assert core.binary_entropy_bits(0.0) < 1e-9 and core.binary_entropy_bits(1.0) < 1e-9
    cue, u, _ = core.draw_noise(400, 120, 20, seed=4)
    o_rand = core.make_outcomes(cue, u, 120, core.phase_probs("random", p=0.5), core.phase_probs("random", p=0.5))
    o_cued = core.make_outcomes(cue, u, 120, core.phase_probs("cued"), core.phase_probs("cued"))
    e_rand = core.leaky_signals(cue, o_rand, 120, 20)["ent"][:, 0].mean()
    e_cued = core.leaky_signals(cue, o_cued, 120, 20)["ent"][:, 0].mean()
    # unlearnable noise keeps outcome uncertainty high; a learnable cue drives it down
    assert e_rand > 0.9 and e_cued < 0.5


def test_associability_update_rule():
    # constant reward after a long run of reward: |PE| small -> associability decays
    S = 4
    cue = np.zeros((S, 40), dtype=np.int8)
    o = np.ones((S, 40), dtype=np.int8)
    sig = core.leaky_signals(cue, o, 20, 20, eta=0.1, assoc0=0.5)
    assert np.all(np.diff(sig["assoc"][0]) <= 1e-12)
    assert sig["assoc"][0, 0] < 0.5


# ---------------------------------------------------------------- change-point model
def test_bocpd_reduces_to_beta_bernoulli_without_change():
    S, n_acq, n_test = 20, 40, 10
    cue, u, _ = core.draw_noise(S, n_acq, n_test, seed=5)
    o = core.make_outcomes(cue, u, n_acq, core.phase_probs("random", p=0.6), core.phase_probs("random", p=0.6))
    sig = core.bocpd_signals(cue, o, n_acq, n_test, h=1e-10)
    for i in range(S):
        counts = []
        for c in (0, 1):
            mask = cue[i, :n_acq] == c
            a = 0.5 + o[i, :n_acq][mask].sum()
            b = 0.5 + (1 - o[i, :n_acq][mask]).sum()
            counts.append(a / (a + b))
        assert abs(sig["V"][i, 0] - np.mean(counts)) < 1e-6


def test_bocpd_detects_change_and_probabilities_valid():
    S, n_acq, n_test = 300, 80, 40
    cue, u, _ = core.draw_noise(S, n_acq, n_test, seed=6)
    o = core.make_outcomes(cue, u, n_acq, core.phase_probs("crf"), core.phase_probs("ext"))
    sig = core.bocpd_signals(cue, o, n_acq, n_test, h=0.02)
    assert sig["V"].min() >= 0 and sig["V"].max() <= 1
    assert sig["pcp"].min() >= 0 and sig["pcp"].max() <= 1 + 1e-9
    assert sig["V"][:, 0].mean() > 0.9           # confident before the change
    assert sig["pcp"][:, 0].mean() < 0.2
    assert sig["pcp"][:, 5].mean() > 0.8         # change detected within a few omissions
    assert sig["V"][:, 5].mean() < 0.6


# ---------------------------------------------------------------- decision rule
def test_quit_times_matches_manual_loop():
    rng = np.random.default_rng(0)
    U = rng.random((50, 30))
    v = rng.random((50, 30))
    c, tau = 0.4, 0.05
    got = core.quit_times(np.hstack([U, U[:, :1]]), v, c, tau)
    for i in range(50):
        q = 30
        for j in range(30):
            if not (v[i, j] < expit((U[i, j] - c) / tau)):
                q = j
                break
        assert got[i] == q


def test_quit_times_monotone_in_threshold_and_deterministic_limit():
    rng = np.random.default_rng(1)
    U = np.sort(rng.random((200, 41)), axis=1)[:, ::-1]  # decreasing decision variable
    v = rng.random((200, 40))
    means = [core.quit_times(U, v, c, 0.02).mean() for c in np.linspace(0.05, 0.95, 10)]
    assert all(np.diff(means) <= 1e-9)
    q0 = core.quit_times(U, v, 0.5, 0.0)
    expected = np.array([np.argmax(~(U[i, :40] > 0.5)) if (~(U[i, :40] > 0.5)).any() else 40 for i in range(200)])
    assert np.array_equal(q0, expected)


def test_calibration_hits_target_and_null():
    rng = np.random.default_rng(2)
    base = np.linspace(0.8, 0.0, 61)[None, :] + 0.02 * rng.standard_normal((500, 61))
    v = rng.random((500, 60))
    c, got, status = core.calibrate_threshold(base, v, 0.02, 15.0)
    assert status == "ok" and abs(got - 15.0) < 0.05
    q, got_null = core.calibrate_null(v, 15.0)
    assert abs(got_null - 15.0) < 0.05
    _, _, status2 = core.calibrate_threshold(base, v, 0.02, 1e6)
    assert status2 == "target_not_bracketed"


def test_equal_magnitude_kappa_scales_mean_magnitude():
    t = np.linspace(0, 1, 40)
    V = np.vstack([0.5 - 0.3 * t] * 5)
    S = np.vstack([0.1 + 0.05 * t] * 5)
    k = core.equal_magnitude_kappa(V, S, window=40, rho=0.5)
    assert np.isclose(np.abs(k * S).mean(), 0.5 * np.abs(V).mean())
    assert core.equal_magnitude_kappa(V, np.zeros_like(S)) == 0.0


def test_lp_traj_matches_inline_and_window_change():
    cue, u, _ = core.draw_noise(50, 60, 30, seed=9)
    o = core.make_outcomes(cue, u, 60, core.phase_probs("cued"), core.phase_probs("cued_rev"))
    sig = core.leaky_signals(cue, o, 60, 30, w=10)
    again = core.lp_traj(sig["loss"], 60, 30, 10)
    assert np.allclose(sig["lp"], again)
    assert not np.allclose(core.lp_traj(sig["loss"], 60, 30, 5), again)


def test_payment_withdrawal_zeroes_extrinsic_term_only():
    cue, u, _ = core.draw_noise(30, 60, 20, seed=10)
    o = core.make_outcomes(cue, u, 60, core.phase_probs("cued"), core.phase_probs("cued"))
    lk = core.leaky_signals(cue, o, 60, 20)
    bc = core.bocpd_signals(cue, o, 60, 20)
    for m in ("R", "CD", "U", "Sp", "IG", "LPs", "LPa", "H", "LPp"):
        V1, S1 = core.model_VS(m, lk, bc, pay=1.0)
        V0, S0 = core.model_VS(m, lk, bc, pay=0.0)
        assert np.all(V0 == 0) and np.array_equal(S0, S1)
        assert V1.max() > 0
    assert np.all(core.model_VS("LPp", lk, bc)[1] >= 0)
