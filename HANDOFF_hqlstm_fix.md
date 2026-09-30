# Handoff: H-QLSTM initialization fix — resume here if chat history is lost

Read this file in full before doing anything else. It is written for a fresh
Claude Code session that has no memory of the conversation that produced it.

## Task

The user (Shankarisr) reported H-QLSTM performing worse than the classical
LSTM and QLSTM baselines on ETTh1 (test RMSE 0.71 vs. 0.63 / 0.59 — see
`results/hqlstm_metrics.json` for whichever numbers are current when you read
this). Instruction: "make hqlstm performance better by doing only logical
changes, dont manipulate anything n get better result" — i.e. find and fix a
real bug/flaw, don't fudge metrics or retune arbitrarily.

## Diagnosis (already done, don't redo)

Root cause: `src/hqlstm_model.py` builds each LSTM gate's quantum circuit as
`qml.ApproxTimeEvolution(H_l, time_l, n=1)` — a single first-order Trotter
step approximating `exp(-i*H*time)`. `qml.qnn.TorchLayer`'s default weight
init draws *every* weight (including `alpha`, `beta`, and `time`) from
`Uniform(0, 2*pi)`. That's correct for a rotation-gate *angle* but wrong for
`time`, which is a duration multiplied by `alpha`/`beta`/`x` before
Trotterization: with all three spanning a full `2*pi` turn independently,
the effective rotation angle at init reached ~250 rad (n_qubits=6), deep in
an aliased/discontinuous regime a single Trotter step badly mis-approximates.
Measured gradient norm at init was large (~26) and unstable (+/-9 std across
random draws), consistent with the observed erratic val-loss oscillation
(0.08 -> 1.28 -> 0.5 -> 1.14 across epochs) and the RMSE regression.

Fix applied in commit `6eca245` ("Fix H-QLSTM's broken VQC weight
initialization..."): custom `init_method` passed to each of the four
`qml.qnn.TorchLayer`s in `HQLSTM.__init__` —
`alpha`/`beta` ~ `Uniform(-1, 1)` (modest O(1) coefficient scale) and `time`
~ `Uniform(0.3, 0.6)` (small sub-1 Trotter-step duration). This range was
chosen by sweeping the *standalone circuit's gradient norm* across
candidate ranges (see git commit message for the numbers), not by looking
at downstream val/test loss — important for defending this as a genuine bug
fix rather than metric manipulation.

**A first attempt at this fix used `time ~ Uniform(0.01, 0.1)`** (too close
to 0) — that collapsed every gate to the `exp(-i*H*0) = identity` fixed
point, where `d/dtheta[cos(theta)] = 0` kills gradients everywhere (verified:
grad norm ~0.18, training stalled at `train_loss~1.0` for two full epochs).
That attempt was reverted before committing. Don't repeat it — if you're
tempted to shrink `time` further, re-derive it from the circuit gradient
sweep, not from intuition.

## What was in progress when this note was written

A retrain of H-QLSTM with the fix was launched in the background and then
**intentionally stopped by the user** (SIGTERM, clean exit) at the end of
epoch 3/30, before logging off for the day — not a crash. Last completed
epoch: `epoch 3/30  train_loss=0.03878  val_loss=0.04469`, already well
below the old (buggy-init) run's best-ever val loss of 0.082, with
train/val tracking together instead of diverging — the loss curve is
stable so far, which is the whole point of the fix. `results/checkpoints/
hqlstm_training_state.pt` reflects this epoch-3 state and is committed at
git commit `<check `git log -- results/checkpoints/hqlstm_training_state.pt`
for the exact hash>`.

**First thing to do: just resume it.** Re-launch the exact same command;
`src/train.py::train_model`'s checkpoint/resume logic will pick up from
epoch 4 automatically using `results/checkpoints/hqlstm_training_state.pt`
(it checks `os.path.exists(checkpoint_path)` and resumes if found — no
flags needed):
```bash
cd /home/student/Desktop/hqlstm && python3 run_hqlstm.py > /tmp/hqlstm_retrain2.log 2>&1
```
Use the Bash tool's `run_in_background: true` directly on this command
(NOT wrapped in `nohup ... &`, which was a mistake made once already in
this task — it makes the tool track the trivial wrapper instead of the
actual long-running process). Expect ~400s/epoch, up to 30 epochs total
with patience=8 early stopping, so up to ~2 hours more from epoch 4.

If for some reason the checkpoint is missing/stale, check whether a
process is still running before assuming it needs restarting:
```bash
ps aux | grep run_hqlstm.py | grep -v grep
tail -50 /tmp/hqlstm_retrain2.log   # may be gone if /tmp was cleared on reboot
cat /home/student/Desktop/hqlstm/results/hqlstm_metrics.json  # only updated when the run FINISHES
```
- If the process is **still running**: just wait / re-check periodically
  (each epoch ~400s). Do not start a second training run in parallel.
- If the process is **gone and `hqlstm_metrics.json` was never updated**
  (check its `epochs_trained` / timestamp against what's described above):
  the run died before finishing. Re-launch it — it will auto-resume from
  `results/checkpoints/hqlstm_training_state.pt` (train.py's checkpoint/
  resume logic in `src/train.py::train_model`), so no progress is lost
  beyond the current epoch in flight:
  ```bash
  cd /home/student/Desktop/hqlstm && python3 run_hqlstm.py > /tmp/hqlstm_retrain3.log 2>&1 &
  ```
  (Better: use the Bash tool's `run_in_background: true` directly on
  `python3 run_hqlstm.py`, NOT wrapped in `nohup ... &` — wrapping it in a
  shell backgrounding operator makes the tool track the trivial wrapper
  command instead of the actual long-running process, which is a mistake
  made once already in this task.)
- If it **finished successfully**: `results/hqlstm_metrics.json` will have
  fresh numbers. Compare `test_rmse_original_units` against the old value
  0.7109 (classical LSTM 0.6316, QLSTM 0.5919 — check
  `results/classical_lstm_metrics.json` / `results/hpo_results.json` or ask
  the user if those files have since changed) to confirm the fix actually
  helped before telling the user it's done.

## After training finishes

1. Verify `results/hqlstm_metrics.json`, `results/checkpoints/hqlstm_best.pt`,
   and the plots in `results/plots/` (`hqlstm_loss_curve.png`,
   `hqlstm_predictions_vs_actual.png`,
   `lstm_vs_qlstm_vs_hqlstm_vs_actual.png`) were regenerated (run_hqlstm.py
   does this automatically at the end of `main()`).
2. Sanity-check the loss curve looks smoother/more stable than the old one
   (train and val loss should move together, not oscillate wildly — that
   was the whole point of the fix).
3. Report the before/after RMSE/MAE/MAPE comparison to the user.
4. Commit the finished results:
   ```bash
   cd /home/student/Desktop/hqlstm
   git add results/hqlstm_metrics.json results/hqlstm_log.txt \
       results/checkpoints/hqlstm_best.pt \
       results/checkpoints/hqlstm_training_state.pt \
       results/plots/hqlstm_loss_curve.png \
       results/plots/hqlstm_predictions_vs_actual.png \
       results/plots/lstm_vs_qlstm_vs_hqlstm_vs_actual.png
   git commit -m "Retrain H-QLSTM with corrected VQC init — results"
   ```
5. Delete this handoff file once the task is confirmed done and committed
   (`git rm HANDOFF_hqlstm_fix.md && git commit -m "Remove stale handoff note"`),
   so it doesn't linger as noise in the repo.

## Key files

- `src/hqlstm_model.py` — the model + the fix (see the long comment above
  `init_method` in `HQLSTM.__init__`)
- `src/qlstm_model.py`, `src/lstm_model.py` — the two baselines being
  compared against (unmodified, capacity-matched via
  `results/model_config.json`)
- `run_hqlstm.py` — training entry point, writes all H-QLSTM artifacts
- `src/train.py` — shared train loop with checkpoint/resume logic (was not
  modified — any grad-clipping or LR changes would need to go here, but
  weren't judged necessary; the init fix alone was enough to stabilize
  training)

## Note on why this file exists

The user asked for these instructions specifically as a fallback in case
this chat's history couldn't be retrieved after a logout/login cycle (their
Claude Code conversation transcripts actually do persist automatically
under `~/.claude/projects/...` and are resumable via the VSCode extension's
"Session history" button or `claude --resume` from a terminal — this file
is a belt-and-suspenders backup, not the primary resume path).
