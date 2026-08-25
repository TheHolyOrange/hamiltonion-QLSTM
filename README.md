# H-QLSTM — Dynamical Hamiltonian Quantum Model for Higher-Order Temporal Dependencies

30% checkpoint: implementation and training of the QLSTM component only, on the
ETTh1 (Electricity Transformer Temperature, hourly) dataset. See
`report/QLSTM_Status_Report.pdf` for full methodology, dataset justification,
hyperparameter search results, and status against project scope.

Reference architecture adapted from
[QCL-PKNU/SPP-QLSTM](https://github.com/QCL-PKNU/SPP-QLSTM)
(Kea et al., *A Hybrid Quantum-Classical Model for Stock Price Prediction Using
Quantum-Enhanced Long Short-Term Memory*, Entropy 2024), generalized here to
arbitrary feature counts, batched quantum circuit execution, and a
train/val/test pipeline with proper scaling and early stopping.

## Defining "higher-order temporal dependencies"

The phrase is used throughout this project's problem statement, so it is
defined precisely here rather than left as a slogan:

- **Statistical definition.** A target y has an *order-p* temporal
  dependency on lagged inputs x_{t-1},...,x_{t-p} if y is independent of
  every strict subset of those lags (no order-<p combination carries any
  information about y), and only the full joint interaction of all p lags
  does. This is the k-wise-independence property of, e.g., the parity
  function, and is the sense in which "higher order" is *not* the same as
  "long lookback window" — a linear function of many lags is still order-1
  in this sense, since each lag contributes independently.
- **Dynamical/Hamiltonian definition** (the one motivating this project's
  architecture). If inputs are encoded into a time-evolution generator,
  U(x,t) = 𝒯 exp(-i∫H(x(t))dt), its Dyson/Magnus expansion produces terms
  such as ∫∫𝒯[H(t₁)H(t₂)]dt₁dt₂, etc.; the n-th order term nonlinearly
  couples n distinct time points via the non-commutativity of H at
  different times. A fixed-ansatz VQC gate recurrence (the current 30%
  checkpoint) is Markovian by construction and does not have algebraic
  access to these terms — this is exactly the gap the planned
  dynamical-Hamiltonian encoding is meant to close.

`src/synthetic_order_experiment.py` tests the statistical definition
directly and controllably: it trains the QLSTM and a classical-LSTM
baseline on order-p parity tasks (p = 1..5, where p-th order parity is
solvable only from the joint interaction of all p lags) and reports
sign-accuracy vs. p (chance = 50%) as `results/synthetic_order_results.json`
and `results/plots/order_dependency_curve.png`. This is a capability probe
against a synthetic, ground-truth-order task, not a claim about ETTh1 — the
ETTh1 multi-scale-periodicity argument in the report motivates dataset
choice but is not itself evidence of order-p capture.

```bash
python3 -m src.synthetic_order_experiment   # writes results/synthetic_order_results.json + plot
```

## Structure

```
data/ETTh1.csv              raw dataset (downloaded from zhouhaoyi/ETDataset)
src/preprocessing.py         loading, cyclical time features, chronological split, scaling, windowing
src/qlstm_model.py           QLSTM cell (4-gate VQC) + regression head
src/train.py                 train/eval loops, early stopping
src/hyperparam_search.py     Optuna search over qubits/layers/hidden size/lr
src/utils.py                 seeding, RMSE/MAE/MAPE
run_pipeline.py               end-to-end: preprocess -> HPO -> final training -> evaluation -> plots
results/                      metrics.json, hpo_results.json, model_config.json, plots/, checkpoints/
report/generate_report.py     builds report/QLSTM_Status_Report.pdf from results/
```

## Reproducing

```bash
pip install -r requirements.txt
python3 run_pipeline.py            # trains QLSTM, writes results/
python3 report/generate_report.py  # builds the PDF status report
```

Key settings (see top of `run_pipeline.py`): 3,500-hour subsample of ETTh1,
24-hour lookback window, 8-trial / 4-epoch Optuna search, final training up to
30 epochs with early stopping (patience 8). These were chosen to keep quantum
circuit simulation (PennyLane `default.qubit`) tractable on CPU for this
checkpoint; see the report's Limitations section for what scaling up implies.
