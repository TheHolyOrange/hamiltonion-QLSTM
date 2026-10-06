"""
Seed-robustness study: retrains H-QLSTM (tuned config) and QLSTM on extra
seeds and records test RMSE per run, so the two models can be compared on
mean and spread across seeds rather than a single seed.

Seed 42 is not rerun here; its results are read from results/hqlstm_metrics.json
(H-QLSTM) and results/metrics.json (QLSTM). Each (model, seed) run has its own
checkpoint file so runs never resume from one another. Re-running the script
skips finished runs and resumes interrupted ones.
"""
import json
import os
import time

import numpy as np
import torch

from src.preprocessing import build_datasets, inverse_transform_target
from src.hqlstm_model import HQLSTMRegressor
from src.qlstm_model import QLSTMRegressor
from src.train import train_model, collect_predictions, make_loaders
from src.utils import set_seed, rmse, resolve_device

SEEDS = [1, 2]
NUM_EPOCHS = 30
PATIENCE = 8
OUT_DIR = "results/seed_study"
CKPT_DIR = "results/checkpoints/seed_study"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    with open("results/model_config.json") as f:
        cfg = json.load(f)
    mc = cfg["model_cfg"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    with open("results/hqlstm_tune_search.json") as f:
        n_trotter = json.load(f)["best_trial"]["n_trotter_steps"]

    train_ds, val_ds, test_ds, meta = build_datasets(
        n_rows=cfg["n_rows"], sequence_length=cfg["sequence_length"])
    num_features = len(meta["feature_cols"])
    scaler, target_idx = meta["scaler"], meta["target_idx"]
    _, _, test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size)

    models = {
        "hqlstm": dict(cls=HQLSTMRegressor,
                       cfg=dict(num_features=num_features, hidden_size=mc["hidden_size"],
                                n_qubits=mc["n_qubits"], n_qlayers=mc["n_qlayers"],
                                n_trotter_steps=n_trotter)),
        "qlstm": dict(cls=QLSTMRegressor,
                      cfg=dict(num_features=num_features, hidden_size=mc["hidden_size"],
                               n_qubits=mc["n_qubits"], n_qlayers=mc["n_qlayers"])),
    }

    for seed in SEEDS:
        for name, spec in models.items():
            out_path = os.path.join(OUT_DIR, f"{name}_seed{seed}.json")
            if os.path.exists(out_path):
                print(f"skip {name} seed {seed} (already done)", flush=True)
                continue

            device = resolve_device(spec["cls"])
            print(f"=== {name} seed {seed} on {device} ===", flush=True)
            set_seed(seed)
            t0 = time.time()
            result = train_model(
                model_cfg=spec["cfg"], train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
                lr=lr, batch_size=batch_size, num_epochs=NUM_EPOCHS, patience=PATIENCE,
                verbose=True, log_fn=lambda m: print(m, flush=True),
                checkpoint_path=os.path.join(CKPT_DIR, f"{name}_seed{seed}_state.pt"),
                device=device, model_cls=spec["cls"],
            )
            train_time = time.time() - t0

            model = result["model"]
            preds_s, actuals_s = collect_predictions(model, test_loader, device=device)
            preds = inverse_transform_target(preds_s, scaler, target_idx, num_features)
            actuals = inverse_transform_target(actuals_s, scaler, target_idx, num_features)
            test_rmse = rmse(actuals, preds)

            record = {
                "model": name, "seed": seed,
                "test_rmse_original_units": test_rmse,
                "best_val_mse_scaled": result["best_val_loss"],
                "epochs_trained": len(result["history"]["train_loss"]),
                "train_time_seconds": train_time,
                "lr": lr, "batch_size": batch_size,
            }
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2)
            print(f"RESULT {name} seed {seed}: test RMSE {test_rmse:.4f}", flush=True)


if __name__ == "__main__":
    main()
