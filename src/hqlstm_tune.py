"""
Dedicated hyperparameter search for H-QLSTM, followed by a full retrain of
the winning config.

run_hqlstm.py (the original H-QLSTM run) deliberately reused QLSTM's
Optuna-selected hyperparameters (results/model_config.json) to isolate the
VQC encoding scheme as the only variable in that comparison. But H-QLSTM's
circuit has a different trainable-parameter geometry than QLSTM's (alpha /
beta / gamma / time vs. QLSTM's layered RX/RY/RZ rotations), a Trotter-step
count that trades numerical fidelity for compute, and a data encoding that
needed its own stability fix (see src/hqlstm_model.py) -- so QLSTM's optimum
hyperparameters have no particular reason to be H-QLSTM's optimum.

This script searches lr x n_trotter_steps at a short training budget (few
epochs, enough to rank configs, not to converge them -- same methodology as
src/hyperparam_search.py), then retrains the winning config for the full
budget used for the other two models (NUM_EPOCHS/PATIENCE below, matching
run_hqlstm.py) and overwrites the canonical results/hqlstm_* artifacts.

n_qubits / n_qlayers / hidden_size / batch_size stay fixed at QLSTM's values
(results/model_config.json) so the three-way comparison remains capacity-
and architecture-scale-matched; only the H-QLSTM-specific knobs are tuned.
"""
import json
import time
import itertools
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

SEARCH_SPACE = {
    "lr": [0.005, 0.01],
    "n_trotter_steps": [1, 2],
}
SEARCH_EPOCHS = 3

NUM_EPOCHS = 30
PATIENCE = 8


def main():
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    with open("results/model_config.json") as f:
        qlstm_cfg = json.load(f)
    hidden_size = qlstm_cfg["model_cfg"]["hidden_size"]
    n_qubits = qlstm_cfg["model_cfg"]["n_qubits"]
    n_qlayers = qlstm_cfg["model_cfg"]["n_qlayers"]
    batch_size = qlstm_cfg["batch_size"]
    sequence_length = qlstm_cfg["sequence_length"]
    n_rows = qlstm_cfg["n_rows"]

    device = resolve_device(HQLSTMRegressor)
    log(f"=== Using device: {device} ===")

    log(f"=== Loading & preprocessing ETTh1 (n_rows={n_rows}, sequence_length={sequence_length}) ===")
    train_ds, val_ds, test_ds, meta = build_datasets(n_rows=n_rows, sequence_length=sequence_length)
    num_features = len(meta["feature_cols"])
    log(f"train/val/test windows: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    # ---- Stage 1: small dedicated search over lr x n_trotter_steps ----
    log(f"\n=== H-QLSTM dedicated search: lr x n_trotter_steps, "
        f"{SEARCH_EPOCHS} epochs/trial, hidden_size={hidden_size}, "
        f"n_qubits={n_qubits}, n_qlayers={n_qlayers} (matched to QLSTM) ===")

    trials = []
    t_search0 = time.time()
    for lr, n_trotter_steps in itertools.product(SEARCH_SPACE["lr"], SEARCH_SPACE["n_trotter_steps"]):
        tag = f"lr={lr},trotter={n_trotter_steps}"
        log(f"\n--- trial {tag} ---")
        model_cfg = dict(num_features=num_features, hidden_size=hidden_size,
                          n_qubits=n_qubits, n_qlayers=n_qlayers,
                          n_trotter_steps=n_trotter_steps)
        set_seed(42)
        t0 = time.time()
        result = train_model(
            model_cfg=model_cfg,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=None,
            lr=lr,
            batch_size=batch_size,
            num_epochs=SEARCH_EPOCHS,
            patience=SEARCH_EPOCHS,  # no early stop during search
            verbose=True,
            log_fn=lambda msg, tag=tag: log(f"[{tag}] {msg}"),
            device=device,
            model_cls=HQLSTMRegressor,
        )
        elapsed = time.time() - t0
        trials.append({"lr": lr, "n_trotter_steps": n_trotter_steps,
                        "best_val_loss": result["best_val_loss"], "elapsed_seconds": elapsed})
        log(f"--- trial {tag} done: best_val_loss={result['best_val_loss']:.5f} ({elapsed:.1f}s) ---")

    search_elapsed = time.time() - t_search0
    best_trial = min(trials, key=lambda t: t["best_val_loss"])
    log(f"\n=== Search complete ({search_elapsed:.1f}s). Best: "
        f"lr={best_trial['lr']}, n_trotter_steps={best_trial['n_trotter_steps']} "
        f"(val_loss={best_trial['best_val_loss']:.5f}) ===")

    with open("results/hqlstm_tune_search.json", "w") as f:
        json.dump({"search_space": SEARCH_SPACE, "search_epochs": SEARCH_EPOCHS,
                    "trials": trials, "best_trial": best_trial,
                    "elapsed_seconds": search_elapsed}, f, indent=2)

    best_lr = best_trial["lr"]
    best_trotter = best_trial["n_trotter_steps"]

    # ---- Stage 2: full retrain of the winning config ----
    model_cfg = dict(num_features=num_features, hidden_size=hidden_size,
                      n_qubits=n_qubits, n_qlayers=n_qlayers,
                      n_trotter_steps=best_trotter)
    log(f"\n=== Full retrain: lr={best_lr}, n_trotter_steps={best_trotter}, "
        f"hidden_size={hidden_size}, n_qubits={n_qubits}, n_qlayers={n_qlayers} ===")
    set_seed(42)
    t0 = time.time()
    result = train_model(
        model_cfg=model_cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        lr=best_lr,
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
        "lr": best_lr,
        "batch_size": batch_size,
        "epochs_trained": len(result["history"]["train_loss"]),
        "n_params": n_params,
        "tuned": True,
        "search_trials": trials,
    }
    log("\n=== H-QLSTM (tuned) test metrics (original OT units, degC) ===")
    log(f"RMSE: {metrics['test_rmse_original_units']:.4f}")
    log(f"MAE:  {metrics['test_mae_original_units']:.4f}")
    log(f"MAPE: {metrics['test_mape_percent']:.2f}%")
    log(f"params: {n_params}")

    with open("results/hqlstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    history = result["history"]
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE (scaled)")
    plt.title("H-QLSTM (tuned) training / validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/hqlstm_loss_curve.png", dpi=150)
    plt.close()

    n_show = min(300, len(preds))
    plt.figure(figsize=(9, 4.5))
    plt.plot(actuals[:n_show], label="actual OT")
    plt.plot(preds[:n_show], label="H-QLSTM (tuned) forecast")
    plt.xlabel("test time step (hours)")
    plt.ylabel("Oil Temperature (degC)")
    plt.title("H-QLSTM (tuned) 1-step-ahead forecast vs actual (test set, first 300 steps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/hqlstm_predictions_vs_actual.png", dpi=150)
    plt.close()

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
        plt.plot(preds[:n_show], label="H-QLSTM (tuned) forecast", alpha=0.85)
        plt.xlabel("test time step (hours)")
        plt.ylabel("Oil Temperature (degC)")
        plt.title("Classical LSTM vs. QLSTM vs. H-QLSTM (tuned) vs. actual (test set, first 300 steps)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/plots/lstm_vs_qlstm_vs_hqlstm_vs_actual.png", dpi=150)
        plt.close()
        log("\nWrote 3-way comparison plot: results/plots/lstm_vs_qlstm_vs_hqlstm_vs_actual.png")
    except (FileNotFoundError, RuntimeError) as e:
        log(f"\nSkipping 3-way comparison plot ({type(e).__name__}: {e}).")

    with open("results/hqlstm_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    log("\n=== H-QLSTM tuning + retrain complete. Artifacts saved to results/ ===")


if __name__ == "__main__":
    main()
