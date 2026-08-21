"""
End-to-end 30%-checkpoint pipeline for the QLSTM component:
  1. Preprocess ETTh1.
  2. Small Optuna hyperparameter search (short trials).
  3. Retrain the best configuration for longer with early stopping.
  4. Evaluate on held-out test set (RMSE / MAE / MAPE, in original OT units).
  5. Save loss curves, prediction-vs-actual plot, metrics.json, model checkpoint.
"""
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.preprocessing import build_datasets, inverse_transform_target
from src.hyperparam_search import run_search
from src.train import train_model, collect_predictions, make_loaders
from src.utils import set_seed, rmse, mae, mape

N_ROWS = 3500
SEQUENCE_LENGTH = 24
HPO_TRIALS = 8
HPO_EPOCHS = 4
FINAL_EPOCHS = 30
FINAL_PATIENCE = 8


def main():
    set_seed(42)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"=== Loading & preprocessing ETTh1 (n_rows={N_ROWS}, sequence_length={SEQUENCE_LENGTH}) ===")
    train_ds, val_ds, test_ds, meta = build_datasets(n_rows=N_ROWS, sequence_length=SEQUENCE_LENGTH)
    num_features = len(meta["feature_cols"])
    log(f"features: {meta['feature_cols']}")
    log(f"train/val/test rows: {meta['train_size']}/{meta['val_size']}/{meta['test_size']}")
    log(f"train/val/test windows: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    log(f"\n=== Hyperparameter search: {HPO_TRIALS} trials x {HPO_EPOCHS} epochs ===")
    t0 = time.time()
    hpo_result = run_search(
        n_trials=HPO_TRIALS,
        search_epochs=HPO_EPOCHS,
        n_rows=N_ROWS,
        sequence_length=SEQUENCE_LENGTH,
        out_path="results/hpo_results.json",
    )
    log(f"HPO elapsed: {time.time()-t0:.1f}s")
    log(f"best params: {hpo_result['best_params']}")

    best_params = hpo_result["best_params"]
    model_cfg = dict(
        num_features=num_features,
        hidden_size=best_params["hidden_size"],
        n_qubits=best_params["n_qubits"],
        n_qlayers=best_params["n_qlayers"],
    )

    log(f"\n=== Final training: up to {FINAL_EPOCHS} epochs, patience={FINAL_PATIENCE} ===")
    set_seed(42)
    t0 = time.time()
    result = train_model(
        model_cfg=model_cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        lr=best_params["lr"],
        batch_size=best_params["batch_size"],
        num_epochs=FINAL_EPOCHS,
        patience=FINAL_PATIENCE,
        verbose=True,
        log_fn=log,
    )
    train_time = time.time() - t0
    log(f"final training elapsed: {train_time:.1f}s")
    log(f"best val loss (scaled): {result['best_val_loss']:.5f}")
    log(f"test loss (scaled MSE): {result['test_loss']:.5f}")

    model = result["model"]
    torch.save(model.state_dict(), "results/checkpoints/qlstm_best.pt")
    with open("results/model_config.json", "w") as f:
        json.dump({"model_cfg": model_cfg, "lr": best_params["lr"],
                   "batch_size": best_params["batch_size"],
                   "sequence_length": SEQUENCE_LENGTH, "n_rows": N_ROWS}, f, indent=2)

    # ---- Evaluate on test set in original units ----
    _, _, test_loader = make_loaders(train_ds, val_ds, test_ds, best_params["batch_size"])
    preds_scaled, actuals_scaled = collect_predictions(model, test_loader)
    scaler = meta["scaler"]
    target_idx = meta["target_idx"]
    preds = inverse_transform_target(preds_scaled, scaler, target_idx, num_features)
    actuals = inverse_transform_target(actuals_scaled, scaler, target_idx, num_features)

    metrics = {
        "test_rmse_original_units": rmse(actuals, preds),
        "test_mae_original_units": mae(actuals, preds),
        "test_mape_percent": mape(actuals, preds),
        "test_mse_scaled": result["test_loss"],
        "best_val_mse_scaled": result["best_val_loss"],
        "train_time_seconds": train_time,
        "model_cfg": model_cfg,
        "lr": best_params["lr"],
        "batch_size": best_params["batch_size"],
        "epochs_trained": len(result["history"]["train_loss"]),
    }
    log(f"\n=== Test metrics (original OT units, degC) ===")
    log(f"RMSE: {metrics['test_rmse_original_units']:.4f}")
    log(f"MAE:  {metrics['test_mae_original_units']:.4f}")
    log(f"MAPE: {metrics['test_mape_percent']:.2f}%")

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Plots ----
    history = result["history"]
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE (scaled)")
    plt.title("QLSTM training / validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    n_show = min(300, len(preds))
    plt.plot(actuals[:n_show], label="actual OT")
    plt.plot(preds[:n_show], label="QLSTM forecast")
    plt.xlabel("test time step (hours)")
    plt.ylabel("Oil Temperature (degC)")
    plt.title("QLSTM 1-step-ahead forecast vs actual (test set, first 300 steps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/predictions_vs_actual.png", dpi=150)
    plt.close()

    with open("results/training_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    log("\n=== Pipeline complete. Artifacts saved to results/ ===")


if __name__ == "__main__":
    main()
