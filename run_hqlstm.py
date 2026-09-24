"""
Hamiltonian-QLSTM (H-QLSTM) on ETTh1, trained/evaluated under the identical
harness used for QLSTM and the classical LSTM baseline (src/train.py,
src/preprocessing.py): same data split, scaling, windowing, optimizer, loss,
and early-stopping protocol.

Reuses the QLSTM's selected hyperparameters (from results/model_config.json,
i.e. the Optuna-selected checkpoint config: hidden_size, n_qubits, n_qlayers,
lr, batch_size) so the three models are capacity-matched -- same as the
classical-LSTM baseline's approach in run_classical_baseline.py. This isolates
the one thing under test: does replacing QLSTM's fixed data-reuploading
ansatz (src/qlstm_model.py) with a dynamical-Hamiltonian generator encoding
(src/hqlstm_model.py, U(x,t) = exp(-i H(x) t)) change forecasting accuracy,
at matched hidden size / qubit count / layer count / data / training budget.

Produces:
  results/hqlstm_metrics.json
  results/checkpoints/hqlstm_best.pt
  results/plots/hqlstm_loss_curve.png
  results/plots/hqlstm_predictions_vs_actual.png
  results/plots/lstm_vs_qlstm_vs_hqlstm_vs_actual.png (3-way comparison, if
    the classical-LSTM and QLSTM checkpoints are present)
"""
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.preprocessing import build_datasets, inverse_transform_target
from src.hqlstm_model import HQLSTMRegressor
from src.qlstm_model import QLSTMRegressor
from src.lstm_model import ClassicalLSTMRegressor
from src.train import train_model, collect_predictions, make_loaders
from src.utils import set_seed, rmse, mae, mape, resolve_device

NUM_EPOCHS = 30
PATIENCE = 8


def main():
    set_seed(42)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    with open("results/model_config.json") as f:
        qlstm_cfg = json.load(f)
    hidden_size = qlstm_cfg["model_cfg"]["hidden_size"]
    n_qubits = qlstm_cfg["model_cfg"]["n_qubits"]
    n_qlayers = qlstm_cfg["model_cfg"]["n_qlayers"]
    lr = qlstm_cfg["lr"]
    batch_size = qlstm_cfg["batch_size"]
    sequence_length = qlstm_cfg["sequence_length"]
    n_rows = qlstm_cfg["n_rows"]

    device = resolve_device(HQLSTMRegressor)
    log(f"=== Using device: {device} (H-QLSTM pins its quantum layers to CPU, "
        f"same as QLSTM; see HQLSTMRegressor.PREFERRED_DEVICE) ===")

    log(f"=== Loading & preprocessing ETTh1 (n_rows={n_rows}, sequence_length={sequence_length}) ===")
    train_ds, val_ds, test_ds, meta = build_datasets(n_rows=n_rows, sequence_length=sequence_length)
    num_features = len(meta["feature_cols"])
    log(f"features: {meta['feature_cols']}")
    log(f"train/val/test windows: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    model_cfg = dict(num_features=num_features, hidden_size=hidden_size,
                      n_qubits=n_qubits, n_qlayers=n_qlayers)
    log(f"\n=== Training H-QLSTM (hidden_size={hidden_size}, n_qubits={n_qubits}, "
        f"n_qlayers={n_qlayers}, matched to QLSTM) ===")
    set_seed(42)
    t0 = time.time()
    result = train_model(
        model_cfg=model_cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        lr=lr,
        batch_size=batch_size,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        verbose=True,
        log_fn=log,
        checkpoint_path="results/checkpoints/hqlstm_training_state.pt",
        device=device,
        model_cls=HQLSTMRegressor,
    )
    train_time = time.time() - t0
    log(f"training elapsed: {train_time:.1f}s")
    log(f"best val loss (scaled): {result['best_val_loss']:.5f}")
    log(f"test loss (scaled MSE): {result['test_loss']:.5f}")

    model = result["model"]
    torch.save(model.state_dict(), "results/checkpoints/hqlstm_best.pt")

    # ---- Evaluate on test set in original units ----
    _, _, test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size)
    preds_scaled, actuals_scaled = collect_predictions(model, test_loader, device=device)
    scaler = meta["scaler"]
    target_idx = meta["target_idx"]
    preds = inverse_transform_target(preds_scaled, scaler, target_idx, num_features)
    actuals = inverse_transform_target(actuals_scaled, scaler, target_idx, num_features)

    n_params = sum(p.numel() for p in model.parameters())
    metrics = {
        "test_rmse_original_units": rmse(actuals, preds),
        "test_mae_original_units": mae(actuals, preds),
        "test_mape_percent": mape(actuals, preds),
        "test_mse_scaled": result["test_loss"],
        "best_val_mse_scaled": result["best_val_loss"],
        "train_time_seconds": train_time,
        "model_cfg": model_cfg,
        "lr": lr,
        "batch_size": batch_size,
        "epochs_trained": len(result["history"]["train_loss"]),
        "n_params": n_params,
    }
    log("\n=== H-QLSTM test metrics (original OT units, degC) ===")
    log(f"RMSE: {metrics['test_rmse_original_units']:.4f}")
    log(f"MAE:  {metrics['test_mae_original_units']:.4f}")
    log(f"MAPE: {metrics['test_mape_percent']:.2f}%")
    log(f"params: {n_params}")

    with open("results/hqlstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Plots ----
    history = result["history"]
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE (scaled)")
    plt.title("H-QLSTM training / validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/hqlstm_loss_curve.png", dpi=150)
    plt.close()

    n_show = min(300, len(preds))
    plt.figure(figsize=(9, 4.5))
    plt.plot(actuals[:n_show], label="actual OT")
    plt.plot(preds[:n_show], label="H-QLSTM forecast")
    plt.xlabel("test time step (hours)")
    plt.ylabel("Oil Temperature (degC)")
    plt.title("H-QLSTM 1-step-ahead forecast vs actual (test set, first 300 steps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/hqlstm_predictions_vs_actual.png", dpi=150)
    plt.close()

    # ---- 3-way comparison plot vs. the already-trained classical LSTM and QLSTM ----
    # Both other models are loaded and run on CPU for this plot regardless of
    # `device`: PennyLane's default.qubit does not reliably keep its internal
    # simulation state on the same device as CUDA input tensors (device-mismatch
    # error in the rotation-gate application), matching run_classical_baseline.py.
    try:
        cpu = torch.device("cpu")
        _, _, cpu_test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size)

        lstm = ClassicalLSTMRegressor(num_features=num_features, hidden_size=hidden_size).to(cpu)
        lstm.load_state_dict(torch.load("results/checkpoints/classical_lstm_best.pt", map_location=cpu))
        lstm.eval()
        lstm_preds_scaled, _ = collect_predictions(lstm, cpu_test_loader, device=cpu)
        lstm_preds = inverse_transform_target(lstm_preds_scaled, scaler, target_idx, num_features)

        qlstm = QLSTMRegressor(**qlstm_cfg["model_cfg"]).to(cpu)
        qlstm.load_state_dict(torch.load("results/checkpoints/qlstm_best.pt", map_location=cpu))
        qlstm.eval()
        q_preds_scaled, _ = collect_predictions(qlstm, cpu_test_loader, device=cpu)
        q_preds = inverse_transform_target(q_preds_scaled, scaler, target_idx, num_features)

        plt.figure(figsize=(9, 4.5))
        plt.plot(actuals[:n_show], label="actual OT", linewidth=1.8)
        plt.plot(lstm_preds[:n_show], label="classical LSTM forecast", alpha=0.85)
        plt.plot(q_preds[:n_show], label="QLSTM forecast", alpha=0.85)
        plt.plot(preds[:n_show], label="H-QLSTM forecast", alpha=0.85)
        plt.xlabel("test time step (hours)")
        plt.ylabel("Oil Temperature (degC)")
        plt.title("Classical LSTM vs. QLSTM vs. H-QLSTM vs. actual (test set, first 300 steps)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/plots/lstm_vs_qlstm_vs_hqlstm_vs_actual.png", dpi=150)
        plt.close()
        log("\nWrote 3-way comparison plot: results/plots/lstm_vs_qlstm_vs_hqlstm_vs_actual.png")
    except (FileNotFoundError, RuntimeError) as e:
        log(f"\nSkipping 3-way comparison plot ({type(e).__name__}: {e}).")

    with open("results/hqlstm_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    log("\n=== H-QLSTM training complete. Artifacts saved to results/ ===")


if __name__ == "__main__":
    main()
