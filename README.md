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
