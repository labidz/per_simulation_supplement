# Persistence after reward withdrawal: simulation code and results

Supplementary material for the manuscript *Persistence after reward withdrawal: dissociating reward
uncertainty, information gain and learning progress* (prepared for *Learning and Motivation*).
Everything here is simulation. No human or animal data are analysed.

## Layout
- `persist/core.py` task streams, shared leaky Beta-Bernoulli learner, Bayesian online change-point model,
  decision signals (reward, change detection, uncertainty, surprise, information gain, learning progress), quit rule.
- `persist/experiments.py` cells, calibration, evaluation, paired-contrast statistics.
- `persist/analysis.py` pre-specified hypothesis checks, signature matrix, plausibility screen.
- `persist/recovery.py` simulated participants, exact quit-time likelihood, parameter/model recovery, power.
- `configs/confirmatory.json` frozen configuration with the six pre-specified predictions
  (SHA-256 7e4ab8fc88fa593011d070f43e9876e3913f125efbdc55a2739c0dc1ad7423e7).
- `scripts/` `pilot.py` (exploratory, seeds 9001/9500, never reused), `run_confirmatory.py`, `run_robustness.py`,
  `run_recovery.py`, `run_recovery_flex.py` (resumable), `check_analytic.py`, `make_figures.py`.
- `tests/` 30 automated tests. `results/` all outputs, including per-seed persistence arrays. `figures/` manuscript figures.
- `.git/` history; tag `confirmatory-freeze` marks the code and configuration frozen before any confirmatory seed was used.

## Reproduce
    pip install -r requirements.txt
    python -m pytest tests -q                                   # 30 tests
    python scripts/run_confirmatory.py                          # about 5 min on one core; writes results/confirmatory_*
    python scripts/run_robustness.py                            # about 6 min
    python scripts/run_recovery.py                              # about 4 min
    MAXSEC=200 python scripts/run_recovery_flex.py              # rerun until it prints a summary (about 11 min)
    python scripts/check_analytic.py && python scripts/make_figures.py

A clean checkout of the tag `confirmatory-freeze` reproduces all 153 persistence arrays, the mean decision-variable
trajectories, the calibrated parameters and the hypothesis table exactly (checked before submission).

## Notes
- Python 3.12, NumPy 2.4, SciPy 1.17, Matplotlib 3.10, pytest 9.
- Code and analyses were written with the assistance of an AI system (see the manuscript's generative-AI declaration);
  review and rerun them before relying on them.
