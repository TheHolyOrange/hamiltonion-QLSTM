"""
Lightweight Optuna hyperparameter search for the QLSTM regressor.

Quantum circuit simulation is the dominant cost, so the search intentionally
uses a small number of trials and a short per-trial training budget (enough
epochs to rank configurations, not to fully converge them). The winning
configuration is then retrained for longer in run_pipeline.py.

The study is persisted to a SQLite file (results/optuna_study.db) so that if
the pipeline is interrupted, re-running it resumes the search instead of
re-running trials that already completed.
"""
import json
import os
import time
import optuna

from src.preprocessing import build_datasets
from src.train import train_model
from src.utils import set_seed

STUDY_NAME = "qlstm_hpo"
STORAGE_PATH = "results/optuna_study.db"

SEARCH_SPACE = {
    "n_qubits": [4, 6],
    "n_qlayers": [1, 2],
    "hidden_size": [8, 16],
    "lr": [0.005, 0.01],
    "batch_size": [32],
}

# Result recovered from the log of the interrupted first run (trial 0 fully
# completed before the process was stopped, but the in-memory-only study of
# that run was lost). Seeded here once so that completed compute is not
# wasted when the persistent study is (re)created from scratch.
_RECOVERED_TRIALS = [
    {"params": {"n_qubits": 6, "n_qlayers": 1, "hidden_size": 8, "lr": 0.01, "batch_size": 32},
     "value": 0.05842746183043346},
]


def objective(trial, train_ds, val_ds, num_features, search_epochs):
    n_qubits = trial.suggest_categorical("n_qubits", SEARCH_SPACE["n_qubits"])
    n_qlayers = trial.suggest_categorical("n_qlayers", SEARCH_SPACE["n_qlayers"])
    hidden_size = trial.suggest_categorical("hidden_size", SEARCH_SPACE["hidden_size"])
    lr = trial.suggest_categorical("lr", SEARCH_SPACE["lr"])
    batch_size = trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"])

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


def _seed_recovered_trials(study):
    distributions = {
        k: optuna.distributions.CategoricalDistribution(v) for k, v in SEARCH_SPACE.items()
    }
    for rec in _RECOVERED_TRIALS:
        study.add_trial(optuna.trial.create_trial(
            params=rec["params"],
            distributions=distributions,
            value=rec["value"],
            state=optuna.trial.TrialState.COMPLETE,
        ))
    print(f"Seeded {len(_RECOVERED_TRIALS)} recovered trial(s) from the interrupted run.")


def run_search(n_trials=8, search_epochs=4, n_rows=3500, sequence_length=24, out_path="results/hpo_results.json"):
    set_seed(42)
    train_ds, val_ds, test_ds, meta = build_datasets(n_rows=n_rows, sequence_length=sequence_length)
    num_features = len(meta["feature_cols"])

    storage_url = f"sqlite:///{STORAGE_PATH}"
    is_new_study = not os.path.exists(STORAGE_PATH)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_url,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    if is_new_study and len(study.trials) == 0:
        _seed_recovered_trials(study)

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)
    print(f"HPO resume state: {completed}/{n_trials} trials already complete, {remaining} remaining.")

    t0 = time.time()
    if remaining > 0:
        study.optimize(
            lambda trial: objective(trial, train_ds, val_ds, num_features, search_epochs),
            n_trials=remaining,
        )
    elapsed = time.time() - t0

    trials_summary = [
        {
            "number": t.number,
            "params": t.params,
            "val_loss": t.value,
        }
        for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
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
    print(f"elapsed (this run): {elapsed:.1f}s")
    return output


if __name__ == "__main__":
    run_search()
