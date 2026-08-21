"""
Lightweight Optuna hyperparameter search for the QLSTM regressor.

Quantum circuit simulation is the dominant cost, so the search intentionally
uses a small number of trials and a short per-trial training budget (enough
epochs to rank configurations, not to fully converge them). The winning
configuration is then retrained for longer in run_pipeline.py.
"""
import json
import time
import optuna

from src.preprocessing import build_datasets
from src.train import train_model
from src.utils import set_seed


def objective(trial, train_ds, val_ds, num_features, search_epochs):
    n_qubits = trial.suggest_categorical("n_qubits", [4, 6])
    n_qlayers = trial.suggest_categorical("n_qlayers", [1, 2])
    hidden_size = trial.suggest_categorical("hidden_size", [8, 16])
    lr = trial.suggest_categorical("lr", [0.005, 0.01])
    batch_size = trial.suggest_categorical("batch_size", [32])

    model_cfg = dict(
        num_features=num_features,
        hidden_size=hidden_size,
        n_qubits=n_qubits,
        n_qlayers=n_qlayers,
    )

    set_seed(42)
    result = train_model(
        model_cfg=model_cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=None,
        lr=lr,
        batch_size=batch_size,
        num_epochs=search_epochs,
        patience=search_epochs,  # no early stop during search, keep it short & fixed
        verbose=True,
        log_fn=lambda msg: print(f"[trial {trial.number}] {msg}"),
    )
    trial.set_user_attr("history", result["history"])
    return result["best_val_loss"]


def run_search(n_trials=8, search_epochs=4, n_rows=3500, sequence_length=24, out_path="results/hpo_results.json"):
    set_seed(42)
    train_ds, val_ds, test_ds, meta = build_datasets(n_rows=n_rows, sequence_length=sequence_length)
    num_features = len(meta["feature_cols"])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(
        lambda trial: objective(trial, train_ds, val_ds, num_features, search_epochs),
        n_trials=n_trials,
    )
    elapsed = time.time() - t0

    trials_summary = [
        {
            "number": t.number,
            "params": t.params,
            "val_loss": t.value,
        }
        for t in study.trials
    ]

    output = {
        "best_params": study.best_params,
        "best_val_loss": study.best_value,
        "n_trials": n_trials,
        "search_epochs": search_epochs,
        "elapsed_seconds": elapsed,
        "trials": trials_summary,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("=== HPO complete ===")
    print(f"best params: {study.best_params}")
    print(f"best val loss: {study.best_value:.5f}")
    print(f"elapsed: {elapsed:.1f}s")
    return output


if __name__ == "__main__":
    run_search()
