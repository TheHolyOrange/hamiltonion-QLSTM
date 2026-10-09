"""
Scratch-only driver: runs just Stage 1 (the lr x n_trotter_steps search) from
src/hqlstm_tune.py, without Stage 2's full retrain. Not part of the repo --
lets us see tuned hyperparameters before committing to the long full-budget
training run. Writes results/hqlstm_tune_search.json, same as the real
script's stage 1 would.
"""
import json
import time
import itertools

from src.preprocessing import build_datasets
from src.hqlstm_model import HQLSTMRegressor
from src.train import train_model
from src.utils import set_seed, resolve_device
from src.hqlstm_tune import SEARCH_SPACE, SEARCH_EPOCHS


def log(msg, log_lines):
    print(msg, flush=True)
    log_lines.append(msg)


def main():
    log_lines = []

    with open("results/model_config.json") as f:
        qlstm_cfg = json.load(f)
    hidden_size = qlstm_cfg["model_cfg"]["hidden_size"]
    n_qubits = qlstm_cfg["model_cfg"]["n_qubits"]
    n_qlayers = qlstm_cfg["model_cfg"]["n_qlayers"]
    batch_size = qlstm_cfg["batch_size"]
    sequence_length = qlstm_cfg["sequence_length"]
    n_rows = qlstm_cfg["n_rows"]

    device = resolve_device(HQLSTMRegressor)
    log(f"=== Using device: {device} ===", log_lines)

    log(f"=== Loading & preprocessing ETTh1 (n_rows={n_rows}, sequence_length={sequence_length}) ===", log_lines)
    train_ds, val_ds, test_ds, meta = build_datasets(n_rows=n_rows, sequence_length=sequence_length)
    num_features = len(meta["feature_cols"])
    log(f"train/val/test windows: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}", log_lines)

    log(f"\n=== H-QLSTM dedicated search: lr x n_trotter_steps, "
        f"{SEARCH_EPOCHS} epochs/trial, hidden_size={hidden_size}, "
        f"n_qubits={n_qubits}, n_qlayers={n_qlayers} (matched to QLSTM) ===", log_lines)

    trials = []
    t_search0 = time.time()
    for lr, n_trotter_steps in itertools.product(SEARCH_SPACE["lr"], SEARCH_SPACE["n_trotter_steps"]):
        tag = f"lr={lr},trotter={n_trotter_steps}"
        log(f"\n--- trial {tag} ---", log_lines)
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
            patience=SEARCH_EPOCHS,
            verbose=True,
            log_fn=lambda msg, tag=tag: log(f"[{tag}] {msg}", log_lines),
            device=device,
            model_cls=HQLSTMRegressor,
        )
        elapsed = time.time() - t0
        trials.append({"lr": lr, "n_trotter_steps": n_trotter_steps,
                        "best_val_loss": result["best_val_loss"], "elapsed_seconds": elapsed})
        log(f"--- trial {tag} done: best_val_loss={result['best_val_loss']:.5f} ({elapsed:.1f}s) ---", log_lines)

    search_elapsed = time.time() - t_search0
    best_trial = min(trials, key=lambda t: t["best_val_loss"])
    log(f"\n=== Search complete ({search_elapsed:.1f}s). Best: "
        f"lr={best_trial['lr']}, n_trotter_steps={best_trial['n_trotter_steps']} "
        f"(val_loss={best_trial['best_val_loss']:.5f}) ===", log_lines)

    with open("results/hqlstm_tune_search.json", "w") as f:
        json.dump({"search_space": SEARCH_SPACE, "search_epochs": SEARCH_EPOCHS,
                    "trials": trials, "best_trial": best_trial,
                    "elapsed_seconds": search_elapsed}, f, indent=2)

    with open("results/hqlstm_search_only_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    log("\n=== Search-only run complete. ===", log_lines)


if __name__ == "__main__":
    main()
