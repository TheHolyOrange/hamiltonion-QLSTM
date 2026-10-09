"""
Train + test the classical LSTM and the QLSTM on the UniEload dataset
(src/unieload_preprocessing.py): next-hour campus load (kW) forecasting.

Both models use the configuration already selected on ETTh1
(results/model_config.json: hidden_size=8, n_qubits=6, n_qlayers=2, lr=0.01,
batch_size=32, 24-hour window) and the same training harness (src/train.py:
Adam, MSE, up to 30 epochs, early stopping patience 8, best-val checkpoint).
No new hyperparameter search is run on UniEload, so the two models are
compared at identical, pre-registered settings rather than each tuned on
this dataset's validation split.

Usage:
    python run_unieload.py lstm
    python run_unieload.py qlstm
    python run_unieload.py compare     # 2-way plot + summary table from saved checkpoints

Artifacts go to results/unieload/ (metrics, logs, plots, checkpoints).
"""
import argparse
import json
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.unieload_preprocessing import build_datasets, inverse_transform_target
from src.lstm_model import ClassicalLSTMRegressor
from src.qlstm_model import QLSTMRegressor
from src.train import train_model, collect_predictions, make_loaders
from src.utils import set_seed, rmse, mae, mape, resolve_device

OUT = "results/unieload"
SEQUENCE_LENGTH = 24
N_ROWS = 3500
NUM_EPOCHS = 30
PATIENCE = 8

with open("results/model_config.json") as _f:
    _ETT_CFG = json.load(_f)
HIDDEN_SIZE = _ETT_CFG["model_cfg"]["hidden_size"]
N_QUBITS = _ETT_CFG["model_cfg"]["n_qubits"]
N_QLAYERS = _ETT_CFG["model_cfg"]["n_qlayers"]
LR = _ETT_CFG["lr"]
BATCH_SIZE = _ETT_CFG["batch_size"]

MODELS = {
    "lstm": dict(cls=ClassicalLSTMRegressor, label="Classical LSTM",
                 cfg=lambda nf: dict(num_features=nf, hidden_size=HIDDEN_SIZE)),
    "qlstm": dict(cls=QLSTMRegressor, label="QLSTM",
                  cfg=lambda nf: dict(num_features=nf, hidden_size=HIDDEN_SIZE,
                                      n_qubits=N_QUBITS, n_qlayers=N_QLAYERS)),
}


def load_data():
    return build_datasets(n_rows=N_ROWS, sequence_length=SEQUENCE_LENGTH)


def load_trained(name, num_features, device=torch.device("cpu")):
    spec = MODELS[name]
    model = spec["cls"](**spec["cfg"](num_features)).to(device)
    model.load_state_dict(torch.load(f"{OUT}/checkpoints/{name}_best.pt", map_location=device))
    model.eval()
    return model


def predict(model, ds, meta, device=torch.device("cpu")):
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    p, a = collect_predictions(model, loader, device=device)
    nf = len(meta["feature_cols"])
    return (inverse_transform_target(p, meta["scaler"], meta["target_idx"], nf),
            inverse_transform_target(a, meta["scaler"], meta["target_idx"], nf))


def train(name):
    spec = MODELS[name]
    os.makedirs(f"{OUT}/checkpoints", exist_ok=True)
    os.makedirs(f"{OUT}/plots", exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    device = resolve_device(spec["cls"])
    train_ds, val_ds, test_ds, meta = load_data()
    nf = len(meta["feature_cols"])
    model_cfg = spec["cfg"](nf)
    log(f"=== UniEload / {spec['label']} on {device} ===")
    log(f"features: {meta['feature_cols']}")
    log(f"rows {meta['date_range']}, splits {meta['split_dates']}")
    log(f"train/val/test windows: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    log(f"model_cfg={model_cfg} lr={LR} batch_size={BATCH_SIZE} epochs<={NUM_EPOCHS} patience={PATIENCE}")

    set_seed(42)
    t0 = time.time()
    result = train_model(
        model_cfg=model_cfg, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        lr=LR, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, patience=PATIENCE,
        verbose=True, log_fn=log,
        checkpoint_path=f"{OUT}/checkpoints/{name}_training_state.pt",
        device=device, model_cls=spec["cls"],
    )
    train_time = time.time() - t0
    model = result["model"]
    torch.save(model.state_dict(), f"{OUT}/checkpoints/{name}_best.pt")

    preds, actuals = predict(model, test_ds, meta, device)
    metrics = {
        "dataset": "UniEload (GSTU_Data_Final_With_Weather.csv)",
        "target": "Load_kW (next hour)",
        "test_rmse_kw": rmse(actuals, preds),
        "test_mae_kw": mae(actuals, preds),
        "test_mape_percent": mape(actuals, preds),
        "test_mse_scaled": result["test_loss"],
        "best_val_mse_scaled": result["best_val_loss"],
        "train_time_seconds": train_time,
        "epochs_trained": len(result["history"]["train_loss"]),
        "n_params": sum(p.numel() for p in model.parameters()),
        "model_cfg": model_cfg, "lr": LR, "batch_size": BATCH_SIZE,
        "history": result["history"],
    }
    log(f"\n=== {spec['label']} test metrics (kW) ===")
    log(f"RMSE: {metrics['test_rmse_kw']:.3f}  MAE: {metrics['test_mae_kw']:.3f}  "
        f"MAPE: {metrics['test_mape_percent']:.2f}%  params: {metrics['n_params']}  "
        f"epochs: {metrics['epochs_trained']}  time: {train_time:.1f}s")
    with open(f"{OUT}/{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    h = result["history"]
    plt.figure(figsize=(7, 4.5))
    plt.plot(h["train_loss"], label="train loss")
    plt.plot(h["val_loss"], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("MSE (scaled)")
    plt.title(f"UniEload - {spec['label']} training / validation loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/plots/{name}_loss_curve.png", dpi=150); plt.close()

    n = min(300, len(preds))
    plt.figure(figsize=(9, 4.5))
    plt.plot(actuals[:n], label="actual load")
    plt.plot(preds[:n], label=f"{spec['label']} forecast")
    plt.xlabel("test time step (hours)"); plt.ylabel("Load (kW)")
    plt.title(f"UniEload - {spec['label']} 1-hour-ahead forecast (test, first {n} h)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/plots/{name}_predictions_vs_actual.png", dpi=150); plt.close()

    with open(f"{OUT}/{name}_log.txt", "w") as f:
        f.write("\n".join(log_lines))


def compare():
    _, _, test_ds, meta = load_data()
    nf = len(meta["feature_cols"])
    rows, preds = [], {}
    for name, spec in MODELS.items():
        with open(f"{OUT}/{name}_metrics.json") as f:
            m = json.load(f)
        rows.append((spec["label"], m))
        preds[name], actuals = predict(load_trained(name, nf), test_ds, meta)

    print(f"{'model':16s} {'RMSE kW':>9s} {'MAE kW':>9s} {'MAPE %':>8s} {'params':>7s} {'epochs':>6s}")
    for label, m in rows:
        print(f"{label:16s} {m['test_rmse_kw']:9.3f} {m['test_mae_kw']:9.3f} "
              f"{m['test_mape_percent']:8.2f} {m['n_params']:7d} {m['epochs_trained']:6d}")

    n = min(300, len(actuals))
    plt.figure(figsize=(9, 4.5))
    plt.plot(actuals[:n], label="actual load", linewidth=1.8)
    for name, spec in MODELS.items():
        plt.plot(preds[name][:n], label=f"{spec['label']} forecast", alpha=0.85)
    plt.xlabel("test time step (hours)"); plt.ylabel("Load (kW)")
    plt.title(f"UniEload - classical LSTM vs. QLSTM vs. actual (test, first {n} h)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/plots/lstm_vs_qlstm_vs_actual.png", dpi=150); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["lstm", "qlstm", "compare"])
    a = ap.parse_args()
    compare() if a.what == "compare" else train(a.what)
