"""
Dedicated hyperparameter search for QLSTM on UniEload, followed by a full
retrain of the winner -- analogous to src/hqlstm_tune.py.

run_unieload.py's QLSTM run reused the ETTh1-selected config verbatim
(lr=0.01, n_qubits=6, n_qlayers=2, hidden_size=8 from results/model_config.json)
for an apples-to-apples baseline comparison against the classical LSTM at
those same settings. That baseline beat the LSTM's own best epoch on epoch 1
(val_loss=0.25953) but then oscillated for 8 more epochs without improving on
it (0.30 -> 0.29 -> 0.27 -> 0.32 -> 0.30 -> 0.27 -> 0.29 -> 0.28) and early-
stopped back at the epoch-1 weights, 1.4% behind the LSTM on test RMSE -- a
config tuned for a different dataset (ETTh1) has no particular reason to be
optimal here, and the oscillation is consistent with lr=0.01 being too large
a step for this VQC on this data.

Quantum circuit simulation is expensive (each UniEload QLSTM epoch cost
roughly 600-2000s wall time in the baseline run), so this search deliberately
tests a short, hand-picked list of configs (not a full grid) at a short
per-trial budget, then retrains only the winner for the full schedule.
Candidates target the two most likely causes of the oscillation:
  - lower lr (0.005, 0.003) at the baseline's capacity (6 qubits, 2 layers),
    to see if a smaller step stabilizes convergence past epoch 1;
  - smaller circuits (4 qubits, or 1 layer) at the baseline lr, which are
    both cheaper per epoch (more epochs feasible in the same wall time) and
    sit further from the barren-plateau regime found in the qubit-count scan
    (results/unieload/analysis/barren_plateau.json shows gradient variance
    falling ~2x per added qubit from n_qubits=4 upward at 2 layers).

The already-completed baseline (lr=0.01, n_qubits=6, n_qlayers=2) is reused
from results/unieload/qlstm_metrics.json's epoch-1 value rather than re-run.

Usage: python -m src.unieload_qlstm_tune
Artifacts: results/unieload/qlstm_tune_search.json (search),
           results/unieload/qlstm_tuned_* (winner's full retrain -- metrics,
           checkpoint, loss curve, prediction plot).
"""
import json
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.unieload_preprocessing import build_datasets, inverse_transform_target
from src.qlstm_model import QLSTMRegressor
from src.lstm_model import ClassicalLSTMRegressor
from src.train import train_model, collect_predictions, make_loaders
from src.utils import set_seed, rmse, mae, mape, resolve_device

OUT = "results/unieload"
SEQUENCE_LENGTH = 24
N_ROWS = 3500
BATCH_SIZE = 32

# Baseline from the already-completed run_unieload.py qlstm run (not re-run):
# lr=0.01, n_qubits=6, n_qlayers=2, hidden_size=8 -> epoch-1 val_loss 0.25953,
# the best epoch of that 9-epoch run (results/unieload/qlstm_log.txt).
BASELINE = dict(label="lr=0.01,nq=6,nl=2 (baseline, already run)",
                 lr=0.01, n_qubits=6, n_qlayers=2, hidden_size=8,
                 best_val_loss=0.25953010376542807, elapsed_seconds=654.4, reused=True)

CANDIDATES = [
    dict(label="lr=0.005,nq=6,nl=2", lr=0.005, n_qubits=6, n_qlayers=2, hidden_size=8),
    dict(label="lr=0.003,nq=6,nl=2", lr=0.003, n_qubits=6, n_qlayers=2, hidden_size=8),
    dict(label="lr=0.01,nq=4,nl=2", lr=0.01, n_qubits=4, n_qlayers=2, hidden_size=8),
    dict(label="lr=0.01,nq=6,nl=1", lr=0.01, n_qubits=6, n_qlayers=1, hidden_size=8),
]
SEARCH_EPOCHS = 4

NUM_EPOCHS = 30
PATIENCE = 8


def main():
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    os.makedirs(f"{OUT}/tune_checkpoints", exist_ok=True)
    device = resolve_device(QLSTMRegressor)
    log(f"=== Using device: {device} ===")

    train_ds, val_ds, test_ds, meta = build_datasets(n_rows=N_ROWS, sequence_length=SEQUENCE_LENGTH)
    num_features = len(meta["feature_cols"])
    log(f"train/val/test windows: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    log(f"\n=== QLSTM search on UniEload: {len(CANDIDATES)} candidates x {SEARCH_EPOCHS} epochs "
        f"(+ 1 reused baseline) ===")
    trials = [dict(BASELINE)]
    log(f"--- {BASELINE['label']}: best_val_loss={BASELINE['best_val_loss']:.5f} (reused, not re-run) ---")

    t_search0 = time.time()
    for cand in CANDIDATES:
        tag = cand["label"]
        log(f"\n--- trial {tag} ---")
        model_cfg = dict(num_features=num_features, hidden_size=cand["hidden_size"],
                          n_qubits=cand["n_qubits"], n_qlayers=cand["n_qlayers"])
        set_seed(42)
        t0 = time.time()
        result = train_model(
            model_cfg=model_cfg, train_ds=train_ds, val_ds=val_ds, test_ds=None,
            lr=cand["lr"], batch_size=BATCH_SIZE, num_epochs=SEARCH_EPOCHS, patience=SEARCH_EPOCHS,
            verbose=True, log_fn=lambda msg, tag=tag: log(f"[{tag}] {msg}"),
            checkpoint_path=f"{OUT}/tune_checkpoints/{tag.replace(',', '_').replace('=', '')}.pt",
            device=device, model_cls=QLSTMRegressor,
        )
        elapsed = time.time() - t0
        trials.append(dict(cand, best_val_loss=result["best_val_loss"], elapsed_seconds=elapsed, reused=False))
        log(f"--- trial {tag} done: best_val_loss={result['best_val_loss']:.5f} ({elapsed:.1f}s) ---")

    search_elapsed = time.time() - t_search0
    best = min(trials, key=lambda t: t["best_val_loss"])
    log(f"\n=== Search complete ({search_elapsed:.1f}s). Best: {best['label']} "
        f"(val_loss={best['best_val_loss']:.5f}) ===")
    with open(f"{OUT}/qlstm_tune_search.json", "w") as f:
        json.dump({"candidates": CANDIDATES, "search_epochs": SEARCH_EPOCHS,
                    "trials": trials, "best_trial": best, "elapsed_seconds": search_elapsed}, f, indent=2)

    if best.get("reused"):
        log("\nBaseline already wins the search; nothing to retrain (see results/unieload/qlstm_metrics.json "
            "for its full 9-epoch run). Exiting without producing qlstm_tuned_* artifacts.")
        with open(f"{OUT}/qlstm_tune_log.txt", "w") as f:
            f.write("\n".join(log_lines))
        return

    # ---- Full retrain of the winning config ----
    model_cfg = dict(num_features=num_features, hidden_size=best["hidden_size"],
                      n_qubits=best["n_qubits"], n_qlayers=best["n_qlayers"])
    log(f"\n=== Full retrain: {best['label']}, lr={best['lr']} ===")
    set_seed(42)
    t0 = time.time()
    result = train_model(
        model_cfg=model_cfg, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        lr=best["lr"], batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, patience=PATIENCE,
        verbose=True, log_fn=log,
        checkpoint_path=f"{OUT}/checkpoints/qlstm_tuned_training_state.pt",
        device=device, model_cls=QLSTMRegressor,
    )
    train_time = time.time() - t0
    model = result["model"]
    torch.save(model.state_dict(), f"{OUT}/checkpoints/qlstm_tuned_best.pt")

    loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    preds_scaled, actuals_scaled = collect_predictions(model, loader, device=device)
    preds = inverse_transform_target(preds_scaled, meta["scaler"], meta["target_idx"], num_features)
    actuals = inverse_transform_target(actuals_scaled, meta["scaler"], meta["target_idx"], num_features)

    n_params = sum(p.numel() for p in model.parameters())
    metrics = {
        "dataset": "UniEload (GSTU_Data_Final_With_Weather.csv)", "target": "Load_kW (next hour)",
        "test_rmse_kw": rmse(actuals, preds), "test_mae_kw": mae(actuals, preds),
        "test_mape_percent": mape(actuals, preds), "test_mse_scaled": result["test_loss"],
        "best_val_mse_scaled": result["best_val_loss"], "train_time_seconds": train_time,
        "epochs_trained": len(result["history"]["train_loss"]), "n_params": n_params,
        "model_cfg": model_cfg, "lr": best["lr"], "batch_size": BATCH_SIZE,
        "tuned": True, "search_trials": trials, "history": result["history"],
    }
    log(f"\n=== QLSTM (tuned) test metrics (kW) ===")
    log(f"RMSE: {metrics['test_rmse_kw']:.3f}  MAE: {metrics['test_mae_kw']:.3f}  "
        f"MAPE: {metrics['test_mape_percent']:.2f}%  params: {n_params}  "
        f"epochs: {metrics['epochs_trained']}  time: {train_time:.1f}s")
    with open(f"{OUT}/qlstm_tuned_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    h = result["history"]
    plt.figure(figsize=(7, 4.5))
    plt.plot(h["train_loss"], label="train loss"); plt.plot(h["val_loss"], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("MSE (scaled)")
    plt.title(f"UniEload - QLSTM (tuned: {best['label']}) loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/plots/qlstm_tuned_loss_curve.png", dpi=150); plt.close()

    n = min(300, len(preds))
    try:
        lstm_cfg = json.load(open(f"{OUT}/lstm_metrics.json"))["model_cfg"]
        lstm = ClassicalLSTMRegressor(**lstm_cfg)
        lstm.load_state_dict(torch.load(f"{OUT}/checkpoints/lstm_best.pt", map_location="cpu"))
        lstm.eval()
        lstm_preds_scaled, _ = collect_predictions(lstm, torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE), device="cpu")
        lstm_preds = inverse_transform_target(lstm_preds_scaled, meta["scaler"], meta["target_idx"], num_features)
        plt.figure(figsize=(9, 4.5))
        plt.plot(actuals[:n], label="actual load", linewidth=1.8)
        plt.plot(lstm_preds[:n], label="Classical LSTM forecast", alpha=0.85)
        plt.plot(preds[:n], label="QLSTM (tuned) forecast", alpha=0.85)
        plt.xlabel("test time step (hours)"); plt.ylabel("Load (kW)")
        plt.title(f"UniEload - LSTM vs. QLSTM (tuned) vs. actual (test, first {n} h)")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{OUT}/plots/lstm_vs_qlstm_tuned_vs_actual.png", dpi=150); plt.close()
    except FileNotFoundError:
        pass

    with open(f"{OUT}/qlstm_tune_log.txt", "w") as f:
        f.write("\n".join(log_lines))
    log("\n=== QLSTM tuning + retrain complete ===")


if __name__ == "__main__":
    main()
