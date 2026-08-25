"""
Controlled synthetic test of "higher-order temporal dependency" capture.

Motivation
----------
The project's stated goal (README, report) is a model that captures
"higher-order temporal dependencies" better than existing hybrid quantum
recurrent designs. Multi-scale periodicity in ETTh1 does not test this claim:
a model can reach low RMSE there with no sensitivity to interaction order at
all. What *does* test it is a task where performance is provably impossible
without access to a joint interaction of a controllable number of lagged
inputs.

Task: order-p parity
---------------------
x_1, ..., x_p are i.i.d. Rademacher (+-1). The target is

    y = x_1 * x_2 * ... * x_p

By construction, for any strict subset S of {x_1,...,x_p} with |S| < p,
y is statistically independent of the values in S (E[y | S] = 0) -- this is
the k-wise independence property of the parity function, and it is the same
combinatorial object as the order-p term in a Dyson/Magnus expansion of a
time-ordered evolution operator: information about the target only appears in
the joint (order-p) interaction of *all* p inputs, never in any lower-order
marginal or pairwise statistic. So "can this architecture fit order-p parity"
is a direct, controllable proxy for "can this architecture represent order-p
temporal interactions," with p swept explicitly rather than asserted.

Each x_i is presented as one time step (window length = p), so recurrence
depth and interaction order p are the same axis by construction: solving
order-p parity requires the recurrent state to retain and nonlinearly combine
all p prior steps, not just the most recent ones.

What this script does
----------------------
For each order p in P_VALUES: build an order-p parity dataset, train the
project's QLSTM and a parameter-comparable classical LSTM baseline under an
identical (small) budget, and record test-set MSE and sign-accuracy for both.
Chance-level sign-accuracy is 50%.

This is a capability probe, not a benchmark result: both architectures are
gate-recurrence / VQC-ansatz based and Markovian in the sense described in
qlstm_model.py's docstring, so neither is expected to have a structural
advantage yet -- that is exactly the gap the planned dynamical-Hamiltonian
encoding (embedding inputs into the generator of time evolution) is intended
to close, since its Dyson-series expansion gives the model direct algebraic
access to order-p multi-time terms instead of requiring gradient descent to
discover them. Results here are the "before" baseline that phase would be
compared against.
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.qlstm_model import QLSTMRegressor
from src.lstm_model import ClassicalLSTMRegressor
from src.utils import set_seed, get_device, resolve_device

P_VALUES = [1, 2, 3, 4, 5]
N_SAMPLES = 3000
HIDDEN_SIZE = 8
N_QUBITS = 4
N_QLAYERS = 1
BATCH_SIZE = 32
NUM_EPOCHS = 20
PATIENCE = 5
LR = 0.01
SEED = 42


def make_parity_dataset(p, n_samples, seed):
    rng = np.random.default_rng(seed)
    x = rng.choice([-1.0, 1.0], size=(n_samples, p)).astype(np.float32)
    y = np.prod(x, axis=1).astype(np.float32)
    X = torch.from_numpy(x).unsqueeze(-1)  # (n_samples, p, 1)
    Y = torch.from_numpy(y)
    return X, Y


def split_dataset(X, Y, train_frac=0.7, val_frac=0.15):
    n = X.shape[0]
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (
        TensorDataset(X[:n_train], Y[:n_train]),
        TensorDataset(X[n_train:n_train + n_val], Y[n_train:n_train + n_val]),
        TensorDataset(X[n_train + n_val:], Y[n_train + n_val:]),
    )


def run_epoch(model, loader, loss_fn, optimizer=None, device=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n_batches = 0.0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in loader:
            if device is not None:
                x, y = x.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, device=None):
    model.eval()
    preds, actuals = [], []
    for x, y in loader:
        if device is not None:
            x, y = x.to(device), y.to(device)
        preds.append(model(x))
        actuals.append(y)
    preds = torch.cat(preds)
    actuals = torch.cat(actuals)
    mse = torch.mean((preds - actuals) ** 2).item()
    acc = torch.mean((torch.sign(preds) == torch.sign(actuals)).float()).item()
    return mse, acc


def train_and_eval(model, train_ds, val_ds, test_ds, log_fn, device):
    device = resolve_device(type(model), device)  # QLSTM pins itself to CPU regardless
    model = model.to(device)
    log_fn(f"    device: {device}")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val, best_state, epochs_no_improve = float("inf"), None, 0
    for epoch in range(NUM_EPOCHS):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device=device)
        val_loss = run_epoch(model, val_loader, loss_fn, optimizer=None, device=device)
        log_fn(f"    epoch {epoch+1}/{NUM_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                log_fn(f"    early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mse, test_acc = evaluate(model, test_loader, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    return {"best_val_mse": best_val, "test_mse": test_mse, "test_acc": test_acc, "n_params": n_params}


def main():
    os.makedirs("results/plots", exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    device = get_device()
    log(f"=== Using device: {device} ===")

    results = {"p_values": P_VALUES, "qlstm": [], "classical_lstm": [], "config": {
        "n_samples": N_SAMPLES, "hidden_size": HIDDEN_SIZE, "n_qubits": N_QUBITS,
        "n_qlayers": N_QLAYERS, "batch_size": BATCH_SIZE, "num_epochs": NUM_EPOCHS,
        "patience": PATIENCE, "lr": LR, "seed": SEED,
    }}

    t0 = time.time()
    for p in P_VALUES:
        log(f"=== order p={p} (window length={p}, n_samples={N_SAMPLES}) ===")
        X, Y = make_parity_dataset(p, N_SAMPLES, seed=SEED + p)
        train_ds, val_ds, test_ds = split_dataset(X, Y)

        set_seed(SEED)
        log("  training QLSTM...")
        qlstm = QLSTMRegressor(num_features=1, hidden_size=HIDDEN_SIZE, n_qubits=N_QUBITS, n_qlayers=N_QLAYERS)
        q_res = train_and_eval(qlstm, train_ds, val_ds, test_ds, log, device)
        q_res["p"] = p
        results["qlstm"].append(q_res)
        log(f"  QLSTM: test_mse={q_res['test_mse']:.4f} test_acc={q_res['test_acc']:.3f} "
            f"n_params={q_res['n_params']}")

        set_seed(SEED)
        log("  training classical LSTM baseline...")
        lstm = ClassicalLSTMRegressor(num_features=1, hidden_size=HIDDEN_SIZE)
        c_res = train_and_eval(lstm, train_ds, val_ds, test_ds, log, device)
        c_res["p"] = p
        results["classical_lstm"].append(c_res)
        log(f"  LSTM:  test_mse={c_res['test_mse']:.4f} test_acc={c_res['test_acc']:.3f} "
            f"n_params={c_res['n_params']}")

    results["elapsed_seconds"] = time.time() - t0
    log(f"\ntotal elapsed: {results['elapsed_seconds']:.1f}s")

    with open("results/synthetic_order_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("results/synthetic_order_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    # --- plot: sign-accuracy vs order p ---
    q_acc = [r["test_acc"] for r in results["qlstm"]]
    c_acc = [r["test_acc"] for r in results["classical_lstm"]]
    plt.figure(figsize=(7, 4.5))
    plt.plot(P_VALUES, q_acc, "o-", label="QLSTM (VQC gates)")
    plt.plot(P_VALUES, c_acc, "s-", label="Classical LSTM")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance level")
    plt.xlabel("dependency order p (parity of p lagged inputs)")
    plt.ylabel("test sign-accuracy")
    plt.ylim(0.35, 1.05)
    plt.xticks(P_VALUES)
    plt.title("Order-p parity: does the architecture capture the full interaction?")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/order_dependency_curve.png", dpi=150)
    plt.close()

    log("\n=== Synthetic order-p experiment complete. Artifacts under results/ ===")


if __name__ == "__main__":
    main()
