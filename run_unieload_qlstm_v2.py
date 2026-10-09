"""
Fast, targeted fix for QLSTM on UniEload, replacing the brute-force
lr/qubit/layer search (src/unieload_qlstm_tune.py, aborted after 2 of 4
candidates: lr=0.005 and lr=0.003 both converged *slower* than the lr=0.01
baseline within a short budget, not to a better optimum -- see
results/unieload/qlstm_tune_search.json) with a direct fix for the actual
failure mode.

Diagnosis (results/unieload/qlstm_metrics.json "history"): with a fixed
lr=0.01, QLSTM's val loss hit a good value on epoch 1 (0.25953) then
oscillated for 8 more epochs without ever beating it (0.30 -> 0.29 -> 0.27 ->
0.32 -> 0.30 -> 0.27 -> 0.29 -> 0.28) -- train loss fell steadily the whole
time, so the model wasn't stuck, but each Adam step at lr=0.01 was large
enough to jump back out of whatever val-loss basin it had just entered. A
smaller *fixed* lr (the aborted search) fixes the step size everywhere,
including early on when a large step is actually fine/helpful, so it just
converges slower; what the oscillation actually calls for is a *schedule*:
keep lr=0.01 for the fast early descent, then cut it once val loss stops
improving so later steps can settle.

Fix (src/train.py, both opt-in / default off elsewhere):
  - use_scheduler=True: ReduceLROnPlateau(factor=0.5, patience=2) on val loss.
  - grad_clip=1.0: clips occasional large gradient spikes (e.g. a VQC
    rotation-angle wraparound) that could otherwise cause one oversized step.
  - patience=12 (vs. the baseline run's 8): gives the scheduler room to cut
    lr at least twice and let training settle at the lower rate before early
    stopping can trigger.
Capacity (n_qubits=6, n_qlayers=2, hidden_size=8) and the starting lr (0.01)
are kept at the baseline's values -- they were never the suspected problem.

Usage: python run_unieload_qlstm_v2.py
Artifacts: results/unieload/qlstm_v2_metrics.json, checkpoints/qlstm_v2_*,
           plots/qlstm_v2_*.
"""
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.unieload_preprocessing import build_datasets, inverse_transform_target
from src.qlstm_model import QLSTMRegressor
from src.lstm_model import ClassicalLSTMRegressor
from src.train import train_model, collect_predictions
from src.utils import set_seed, rmse, mae, mape, resolve_device

OUT = "results/unieload"
SEQUENCE_LENGTH = 24
N_ROWS = 3500
BATCH_SIZE = 32
LR = 0.01
NUM_EPOCHS = 30
PATIENCE = 12
GRAD_CLIP = 1.0

with open("results/model_config.json") as _f:
    _ETT_CFG = json.load(_f)
MODEL_CFG_BASE = dict(hidden_size=_ETT_CFG["model_cfg"]["hidden_size"],
                       n_qubits=_ETT_CFG["model_cfg"]["n_qubits"],
                       n_qlayers=_ETT_CFG["model_cfg"]["n_qlayers"])


def main():
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    device = resolve_device(QLSTMRegressor)
    train_ds, val_ds, test_ds, meta = build_datasets(n_rows=N_ROWS, sequence_length=SEQUENCE_LENGTH)
    num_features = len(meta["feature_cols"])
    model_cfg = dict(num_features=num_features, **MODEL_CFG_BASE)
    log(f"=== UniEload / QLSTM v2 (scheduler + grad clip) on {device} ===")
    log(f"model_cfg={model_cfg} lr0={LR} batch_size={BATCH_SIZE} epochs<={NUM_EPOCHS} "
        f"patience={PATIENCE} grad_clip={GRAD_CLIP} scheduler=ReduceLROnPlateau(factor=0.5,patience=2)")
    log(f"train/val/test windows: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    set_seed(42)
    t0 = time.time()
    result = train_model(
        model_cfg=model_cfg, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        lr=LR, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, patience=PATIENCE,
        verbose=True, log_fn=log,
        checkpoint_path=f"{OUT}/checkpoints/qlstm_v2_training_state.pt",
        device=device, model_cls=QLSTMRegressor,
        grad_clip=GRAD_CLIP, use_scheduler=True, scheduler_factor=0.5, scheduler_patience=2,
    )
    train_time = time.time() - t0
    model = result["model"]
    torch.save(model.state_dict(), f"{OUT}/checkpoints/qlstm_v2_best.pt")

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
        "model_cfg": model_cfg, "lr0": LR, "batch_size": BATCH_SIZE, "grad_clip": GRAD_CLIP,
        "scheduler": "ReduceLROnPlateau(factor=0.5,patience=2)",
        "fix": "lr-on-plateau schedule + grad clipping (vs. fixed lr=0.01 baseline)",
        "history": result["history"],
    }
    log(f"\n=== QLSTM v2 test metrics (kW) ===")
    log(f"RMSE: {metrics['test_rmse_kw']:.3f}  MAE: {metrics['test_mae_kw']:.3f}  "
        f"MAPE: {metrics['test_mape_percent']:.2f}%  params: {n_params}  "
        f"epochs: {metrics['epochs_trained']}  time: {train_time:.1f}s")
    with open(f"{OUT}/qlstm_v2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    h = result["history"]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(h["train_loss"], label="train loss"); ax1.plot(h["val_loss"], label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("MSE (scaled)")
    ax2 = ax1.twinx()
    ax2.plot(h["lr"], "k--", alpha=0.5, label="lr")
    ax2.set_ylabel("learning rate"); ax2.set_yscale("log")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("UniEload - QLSTM v2 (lr-on-plateau) loss + lr schedule")
    plt.tight_layout()
    plt.savefig(f"{OUT}/plots/qlstm_v2_loss_curve.png", dpi=150); plt.close()

    n = min(300, len(preds))
    try:
        lstm_cfg = json.load(open(f"{OUT}/lstm_metrics.json"))["model_cfg"]
        lstm = ClassicalLSTMRegressor(**lstm_cfg)
        lstm.load_state_dict(torch.load(f"{OUT}/checkpoints/lstm_best.pt", map_location="cpu"))
        lstm.eval()
        lstm_preds_scaled, _ = collect_predictions(
            lstm, torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE), device="cpu")
        lstm_preds = inverse_transform_target(lstm_preds_scaled, meta["scaler"], meta["target_idx"], num_features)
        plt.figure(figsize=(9, 4.5))
        plt.plot(actuals[:n], label="actual load", linewidth=1.8)
        plt.plot(lstm_preds[:n], label="Classical LSTM forecast", alpha=0.85)
        plt.plot(preds[:n], label="QLSTM v2 forecast", alpha=0.85)
        plt.xlabel("test time step (hours)"); plt.ylabel("Load (kW)")
        plt.title(f"UniEload - LSTM vs. QLSTM v2 vs. actual (test, first {n} h)")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{OUT}/plots/lstm_vs_qlstm_v2_vs_actual.png", dpi=150); plt.close()
    except FileNotFoundError:
        pass

    with open(f"{OUT}/qlstm_v2_log.txt", "w") as f:
        f.write("\n".join(log_lines))
    log("\n=== QLSTM v2 complete ===")


if __name__ == "__main__":
    main()
