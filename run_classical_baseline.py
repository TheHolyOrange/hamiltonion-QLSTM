"""
Classical LSTM baseline on ETTh1, trained/evaluated under the identical harness
used for the QLSTM (src/train.py, src/preprocessing.py): same data split,
scaling, windowing, optimizer, loss, and early-stopping protocol.

Uses the QLSTM's selected hidden_size (from results/model_config.json, i.e.
the Optuna-selected checkpoint config) so the two models are capacity-matched
on hidden state size; no separate HPO is run for the classical LSTM since,
unlike the VQC gates, a plain nn.LSTM at this scale is cheap and not the
object of the comparison -- the comparison is "does the VQC gate recurrence
buy anything over a classical gate recurrence of the same hidden size."

Produces:
  results/classical_lstm_metrics.json
  results/checkpoints/classical_lstm_best.pt
  results/plots/classical_lstm_loss_curve.png
  results/plots/classical_lstm_predictions_vs_actual.png
  results/plots/qlstm_vs_classical_vs_actual.png   (comparison, if QLSTM checkpoint present)
"""
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.preprocessing import build_datasets, inverse_transform_target
from src.lstm_model import ClassicalLSTMRegressor
from src.qlstm_model import QLSTMRegressor
from src.train import train_model, collect_predictions, make_loaders
from src.utils import set_seed, rmse, mae, mape, get_device

N_ROWS = 3500
SEQUENCE_LENGTH = 24
NUM_EPOCHS = 30
PATIENCE = 8


def main():
    set_seed(42)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    device = get_device()
    log(f"=== Using device: {device} ===")

    with open("results/model_config.json") as f:
        qlstm_cfg = json.load(f)
    hidden_size = qlstm_cfg["model_cfg"]["hidden_size"]
    lr = qlstm_cfg["lr"]
    batch_size = qlstm_cfg["batch_size"]

    log(f"=== Loading & preprocessing ETTh1 (n_rows={N_ROWS}, sequence_length={SEQUENCE_LENGTH}) ===")
    train_ds, val_ds, test_ds, meta = build_datasets(n_rows=N_ROWS, sequence_length=SEQUENCE_LENGTH)
    num_features = len(meta["feature_cols"])
    log(f"features: {meta['feature_cols']}")
    log(f"train/val/test windows: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    model_cfg = dict(num_features=num_features, hidden_size=hidden_size)
    log(f"\n=== Training classical LSTM baseline (hidden_size={hidden_size}, matched to QLSTM) ===")
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
        checkpoint_path="results/checkpoints/classical_lstm_training_state.pt",
        device=device,
        model_cls=ClassicalLSTMRegressor,
    )
    train_time = time.time() - t0
    log(f"training elapsed: {train_time:.1f}s")
    log(f"best val loss (scaled): {result['best_val_loss']:.5f}")
    log(f"test loss (scaled MSE): {result['test_loss']:.5f}")

    model = result["model"]
    torch.save(model.state_dict(), "results/checkpoints/classical_lstm_best.pt")

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
    log("\n=== Classical LSTM test metrics (original OT units, degC) ===")
    log(f"RMSE: {metrics['test_rmse_original_units']:.4f}")
    log(f"MAE:  {metrics['test_mae_original_units']:.4f}")
    log(f"MAPE: {metrics['test_mape_percent']:.2f}%")
    log(f"params: {n_params}")

    with open("results/classical_lstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Plots ----
    history = result["history"]
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE (scaled)")
    plt.title("Classical LSTM training / validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/classical_lstm_loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    n_show = min(300, len(preds))
    plt.plot(actuals[:n_show], label="actual OT")
    plt.plot(preds[:n_show], label="classical LSTM forecast")
    plt.xlabel("test time step (hours)")
    plt.ylabel("Oil Temperature (degC)")
    plt.title("Classical LSTM 1-step-ahead forecast vs actual (test set, first 300 steps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/classical_lstm_predictions_vs_actual.png", dpi=150)
    plt.close()

    # ---- Combined comparison plot vs. the already-trained QLSTM, if available ----
    # PennyLane's default.qubit does not reliably keep its internal simulation
    # state on the same device as CUDA input tensors (device-mismatch error in
    # the rotation-gate application), so the QLSTM is loaded and run on CPU
    # here regardless of what device training used.
    try:
        cpu = torch.device("cpu")
        qlstm = QLSTMRegressor(**qlstm_cfg["model_cfg"]).to(cpu)
        qlstm.load_state_dict(torch.load("results/checkpoints/qlstm_best.pt", map_location=cpu))
        qlstm.eval()
        _, _, cpu_test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size)
        q_preds_scaled, _ = collect_predictions(qlstm, cpu_test_loader, device=cpu)
        q_preds = inverse_transform_target(q_preds_scaled, scaler, target_idx, num_features)

        plt.figure(figsize=(9, 4.5))
        plt.plot(actuals[:n_show], label="actual OT", linewidth=1.8)
        plt.plot(q_preds[:n_show], label="QLSTM forecast", alpha=0.85)
        plt.plot(preds[:n_show], label="classical LSTM forecast", alpha=0.85)
        plt.xlabel("test time step (hours)")
        plt.ylabel("Oil Temperature (degC)")
        plt.title("QLSTM vs. classical LSTM vs. actual (test set, first 300 steps)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/plots/qlstm_vs_classical_vs_actual.png", dpi=150)
        plt.close()
        log("\nWrote comparison plot: results/plots/qlstm_vs_classical_vs_actual.png")
    except (FileNotFoundError, RuntimeError) as e:
        log(f"\nSkipping QLSTM comparison plot ({type(e).__name__}: {e}).")

    with open("results/classical_lstm_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    log("\n=== Classical LSTM baseline complete. Artifacts saved to results/ ===")


if __name__ == "__main__":
    main()
