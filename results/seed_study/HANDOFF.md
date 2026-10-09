# Seed study handoff

## Goal
Decide whether H-QLSTM (tuned) beats QLSTM. Seed 42 alone gives a test-RMSE gap of
0.5785 (H-QLSTM) vs 0.5919 (QLSTM), which a bootstrap says is within noise.
The seed study retrains both models on seeds 1 and 2 and compares them across seeds.

## Current state (paused)
- Script: `run_seed_study.py` (repo root)
- Finished: seed 42 for both models (`results/hqlstm_metrics.json`, `results/metrics.json`)
- In progress: `hqlstm` seed 1, checkpointed after completed epoch 9
  (best val 0.05066 at that point; the best overall is still seed 42's 0.03980 at epoch 18)
- Not started: `hqlstm` seed 2, `qlstm` seeds 1 and 2

## How to resume
From the repo root, with the same Python environment:

    python -u run_seed_study.py >> results/seed_study/run_console.log 2>&1

It resumes `hqlstm` seed 1 from its checkpoint, then runs the remaining models in order.
Finished runs are skipped; each writes `results/seed_study/<model>_seed<seed>.json`.

## Timing
- H-QLSTM: about 8.5 minutes per epoch on CPU, up to 30 epochs, early stopping patience 8.
  Each run takes roughly 2 to 3 hours.
- QLSTM: about 50 minutes per run.
- Remaining work is roughly 5 to 7 hours.

## Caveat
The epoch-9 timing in the log shows 44443 s. That reflects a machine suspend during the run,
not compute time. Real epoch time was about 510 s.

## When done
Compare test RMSE across seeds 42, 1, and 2 for each model (mean and spread).
Only then decide whether H-QLSTM's advantage over QLSTM is real.
