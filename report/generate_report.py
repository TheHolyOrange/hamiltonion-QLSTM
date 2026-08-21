"""
Generates the status report PDF for the 30% implementation checkpoint of the
'Dynamical Hamiltonian Quantum Model for Higher Order Temporal Dependencies'
project (QLSTM component only).

Reads: results/metrics.json, results/hpo_results.json, results/model_config.json,
       results/plots/loss_curve.png, results/plots/predictions_vs_actual.png
Writes: report/QLSTM_Status_Report.pdf
"""
import json
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, HRFlowable
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(rel_path):
    with open(os.path.join(ROOT, rel_path)) as f:
        return json.load(f)


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1c", parent=styles["Heading1"], spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#1a2b4c")))
    styles.add(ParagraphStyle(name="H2c", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#2c4a7c")))
    styles.add(ParagraphStyle(name="Bodyc", parent=styles["BodyText"], spaceAfter=6, leading=14))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11, textColor=colors.HexColor("#444444")))
    styles.add(ParagraphStyle(name="TitleBig", parent=styles["Title"], fontSize=20, spaceAfter=4))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontSize=12, textColor=colors.HexColor("#555555"), spaceAfter=20, alignment=1))
    return styles


def metric_table(metrics, styles):
    data = [
        ["Metric", "Value"],
        ["Test RMSE (original units, degC)", f"{metrics['test_rmse_original_units']:.4f}"],
        ["Test MAE (original units, degC)", f"{metrics['test_mae_original_units']:.4f}"],
        ["Test MAPE", f"{metrics['test_mape_percent']:.2f}%"],
        ["Test MSE (scaled)", f"{metrics['test_mse_scaled']:.5f}"],
        ["Best validation MSE (scaled)", f"{metrics['best_val_mse_scaled']:.5f}"],
        ["Epochs trained (early stop)", f"{metrics['epochs_trained']}"],
        ["Training wall-clock time", f"{metrics['train_time_seconds']:.1f} s"],
    ]
    t = Table(data, colWidths=[9 * cm, 6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2b4c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def config_table(model_cfg, extra, styles):
    data = [["Hyperparameter", "Value"]]
    labels = {
        "num_features": "Input feature count",
        "hidden_size": "Hidden size",
        "n_qubits": "Number of qubits (per gate)",
        "n_qlayers": "Variational layers (VQC depth)",
    }
    for k, v in model_cfg.items():
        data.append([labels.get(k, k), str(v)])
    for k, v in extra.items():
        data.append([k, str(v)])
    t = Table(data, colWidths=[9 * cm, 6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2b4c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def hpo_table(trials, styles):
    header = ["#", "n_qubits", "n_qlayers", "hidden", "lr", "val MSE (scaled)"]
    data = [header]
    trials_sorted = sorted(trials, key=lambda t: t["val_loss"])
    for t in trials_sorted:
        p = t["params"]
        data.append([
            str(t["number"]), str(p["n_qubits"]), str(p["n_qlayers"]),
            str(p["hidden_size"]), str(p["lr"]), f"{t['val_loss']:.5f}"
        ])
    tbl = Table(data, colWidths=[1.2 * cm, 2.3 * cm, 2.3 * cm, 2 * cm, 2 * cm, 3.7 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2b4c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#d9e6c8")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def bullets(items, styles, style_name="Bodyc"):
    return ListFlowable(
        [ListItem(Paragraph(it, styles[style_name]), bulletColor=colors.HexColor("#2c4a7c")) for it in items],
        bulletType="bullet", start="•", leftIndent=14,
    )


def main():
    metrics = load_json("results/metrics.json")
    hpo = load_json("results/hpo_results.json")
    model_config = load_json("results/model_config.json")

    styles = build_styles()
    doc = SimpleDocTemplate(
        os.path.join(ROOT, "report", "QLSTM_Status_Report.pdf"),
        pagesize=A4,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
        title="QLSTM Implementation Status Report",
        author="H-QLSTM Project",
    )

    story = []

    # --- Title page ---
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Dynamical Hamiltonian Quantum Model for", styles["TitleBig"]))
    story.append(Paragraph("Higher-Order Temporal Dependencies", styles["TitleBig"]))
    story.append(Paragraph("QLSTM Component — 30% Implementation Status Report", styles["Subtitle"]))
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#1a2b4c"), thickness=1))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(
        "This report documents the current implementation status of the quantum-enhanced "
        "LSTM (QLSTM) component of the project. Per the review scope, only the QLSTM model "
        "has been implemented and trained at this stage; the dynamical-Hamiltonian data "
        "encoding, the full comparative benchmarking against classical and variational-quantum "
        "baselines, and large-scale hyperparameter optimization are planned for subsequent phases.",
        styles["Bodyc"]))
    story.append(Spacer(1, 2 * cm))
    story.append(PageBreak())

    # --- 1. Problem statement ---
    story.append(Paragraph("1. Problem Statement", styles["H1c"]))
    story.append(Paragraph(
        "Classical and current hybrid quantum recurrent models struggle to efficiently capture "
        "higher-order temporal dependencies in complex nonlinear time-series due to limited "
        "representational capacity and shallow data encodings. Quantum LSTM (QLSTM) frameworks "
        "have shown promise in exploiting quantum feature spaces for sequence learning, but "
        "existing designs primarily embed data via variational circuits without explicit "
        "Hamiltonian-driven dynamics to model continuous temporal evolution. This project aims "
        "to design, optimize, and benchmark a dynamical-Hamiltonian-encoded QLSTM architecture "
        "against classical and variational-quantum baselines. This report covers the first "
        "milestone: implementing and training the baseline VQC-based QLSTM cell that will later "
        "be extended with an explicit Hamiltonian encoding.", styles["Bodyc"]))

    story.append(Paragraph("2. Reference Implementation Used", styles["H1c"]))
    story.append(Paragraph(
        "The implementation is adapted from the official code of Kea et al., "
        "\"A Hybrid Quantum-Classical Model for Stock Price Prediction Using Quantum-Enhanced "
        "Long Short-Term Memory\" (Entropy, 2024) — repository "
        "<b>QCL-PKNU/SPP-QLSTM</b>. The reference QLSTM cell replaces each of the four LSTM "
        "gates (forget, input, candidate/update, output) with a variational quantum circuit "
        "(VQC) built from angle embedding, an entangling layer of CNOTs, and RX/RY/RZ rotations, "
        "implemented in PennyLane with a PyTorch interface. The reference code targets a single "
        "univariate stock-price series (AAPL daily close, ~250 rows) with batch size 1 and a "
        "fixed 3-step sequence length.", styles["Bodyc"]))
    story.append(Paragraph(
        "The reference implementation is dataset-specific (single feature, tiny sample count, "
        "no validation split, unbatched training). For this project the QLSTM architecture was "
        "re-implemented to be dataset-agnostic: generalized to arbitrary feature counts, "
        "batched circuit execution (PennyLane parameter broadcasting), a proper "
        "train/validation/test split, feature scaling, and early stopping.", styles["Bodyc"]))

    # --- 3. Dataset selection ---
    story.append(Paragraph("3. Dataset Selection", styles["H1c"]))
    story.append(Paragraph(
        "Three candidate datasets were specified: the ETT (Electricity Transformer Temperature) "
        "dataset, the UCI Electricity Load Diagrams dataset, and the Jena Climate dataset. "
        "<b>ETTh1</b> (hourly-resolution ETT) was selected for this checkpoint:", styles["Bodyc"]))
    story.append(bullets([
        "Right-sized for quantum circuit simulation: 17,420 hourly rows with 6 load features "
        "plus an oil-temperature (OT) target, versus UCI Electricity Load's 370 concurrent "
        "meters at 15-min resolution (~140k rows per meter) and Jena Climate's ~420k rows at "
        "10-min resolution — both would require heavy subsampling before a state-vector "
        "quantum simulator becomes tractable, whereas ETTh1 is usable close to its native form.",
        "Established long-sequence forecasting benchmark (used in Informer/Autoformer and "
        "related literature), which makes result comparisons and future baseline benchmarking "
        "straightforward.",
        "Exhibits clear multi-scale periodicity (diurnal load cycles, weekly patterns overlaid "
        "on trend), which directly exercises the \"higher-order temporal dependencies\" this "
        "project targets.",
        "Single, directly downloadable CSV with no license/access friction (unlike UCI's "
        "semicolon/decimal-comma formatted archive or Jena's session-based portal).",
    ], styles))

    # --- 4. Preprocessing ---
    story.append(Paragraph("4. Data Preprocessing Pipeline", styles["H1c"]))
    story.append(bullets([
        f"<b>Subsampling:</b> the most recent {model_config['n_rows']} hourly rows were used "
        "(quantum circuit simulation cost scales with the number of training windows; this is "
        "a scoping decision for the 30% checkpoint, not a modeling limitation — see Section 8).",
        "<b>Feature engineering:</b> the 6 raw load features (HUFL, HULL, MUFL, MULL, LUFL, LULL) "
        "and the OT target are kept, plus 4 cyclical time features "
        "(sin/cos of hour-of-day, sin/cos of day-of-week) so the model has explicit access to "
        "periodic structure rather than having to infer it purely from the recurrence.",
        "<b>Chronological split:</b> 70% train / 15% validation / 15% test, split by time order "
        "(no shuffling across the split boundary) to avoid look-ahead leakage.",
        "<b>Scaling:</b> StandardScaler fit on the training split only, applied to validation "
        "and test splits (fit only on train to prevent test-set leakage).",
        f"<b>Windowing:</b> sliding windows of length {model_config['sequence_length']} hours "
        "(one full day of lookback) are used to predict the OT value at the next hour "
        "(1-step-ahead forecasting).",
    ], styles))

    # --- 5. Architecture ---
    story.append(Paragraph("5. QLSTM Architecture", styles["H1c"]))
    story.append(Paragraph(
        "At each time step t, the concatenation of the previous hidden state h<sub>t-1</sub> and "
        "the current input x<sub>t</sub> is linearly projected down to n_qubits dimensions, then "
        "passed through four independent variational quantum circuits (one per LSTM gate). Each "
        "VQC applies angle embedding of the classical features into qubit rotations, an "
        "entangling layer of CNOTs, and a trainable RX/RY/RZ rotation block, repeated for "
        "n_qlayers layers. The Pauli-Z expectation values are read out and linearly projected "
        "back to hidden_size, followed by the standard LSTM gate nonlinearities and the standard "
        "cell/hidden state update:", styles["Bodyc"]))
    story.append(Paragraph(
        "c<sub>t</sub> = f<sub>t</sub> &middot; c<sub>t-1</sub> + i<sub>t</sub> &middot; g<sub>t</sub> "
        "&nbsp;&nbsp;&nbsp; h<sub>t</sub> = o<sub>t</sub> &middot; tanh(c<sub>t</sub>)",
        styles["Bodyc"]))
    story.append(Paragraph(
        "A final linear head maps the last hidden state to the single-step OT forecast. "
        "<i>Note:</i> this is the classical-VQC-hybrid baseline. The project's proposed "
        "dynamical-Hamiltonian encoding — embedding the input into the generator of time "
        "evolution (e.g. U(x,t) = exp(-i H(x) t)) rather than into a fixed data-reuploading "
        "ansatz — is the planned extension for the next phase and is not yet implemented.",
        styles["Bodyc"]))

    story.append(Paragraph("5.1 Selected Configuration (from hyperparameter search)", styles["H2c"]))
    story.append(config_table(metrics["model_cfg"], {
        "Learning rate": metrics["lr"],
        "Batch size": metrics["batch_size"],
        "Sequence length": model_config["sequence_length"],
        "Rows used": model_config["n_rows"],
    }, styles))

    story.append(PageBreak())

    # --- 6. HPO ---
    story.append(Paragraph("6. Hyperparameter Search", styles["H1c"]))
    story.append(Paragraph(
        f"An Optuna (TPE sampler) search was run over {hpo['n_trials']} trials, each trained "
        f"for {hpo['search_epochs']} epochs (a short budget sufficient to rank configurations "
        "given the cost of quantum circuit simulation; the winning configuration was then "
        "retrained for longer — see Section 7). Search space: n_qubits &isin; {4, 6}, "
        "n_qlayers &isin; {1, 2}, hidden_size &isin; {8, 16}, learning_rate &isin; "
        "{0.005, 0.01}, batch_size = 32.", styles["Bodyc"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(hpo_table(hpo["trials"], styles))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        f"Search wall-clock time: {hpo['elapsed_seconds']:.1f}s. Best configuration highlighted "
        "above was carried forward to final training.", styles["Small"]))

    # --- 7. Training & results ---
    story.append(Paragraph("7. Final Training & Test Results", styles["H1c"]))
    story.append(Paragraph(
        "The best configuration from the search was retrained with early stopping "
        "(monitoring validation MSE, patience epochs as configured in run_pipeline.py) using "
        "the Adam optimizer and MSE loss.", styles["Bodyc"]))
    story.append(metric_table(metrics, styles))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Image(os.path.join(ROOT, "results/plots/loss_curve.png"), width=15 * cm, height=9.6 * cm))
    story.append(Paragraph("Figure 1: Training and validation MSE loss (scaled units) per epoch.", styles["Small"]))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Image(os.path.join(ROOT, "results/plots/predictions_vs_actual.png"), width=15 * cm, height=7.5 * cm))
    story.append(Paragraph("Figure 2: QLSTM 1-step-ahead forecast vs. actual oil temperature on held-out test data "
                            "(first 300 hourly steps shown, original units).", styles["Small"]))

    story.append(PageBreak())

    # --- 8. Status vs scope ---
    story.append(Paragraph("8. Implementation Status vs. Project Scope", styles["H1c"]))
    story.append(Paragraph("<b>Completed in this checkpoint (~30%):</b>", styles["Bodyc"]))
    story.append(bullets([
        "Dataset selection and justification (ETTh1) against the three candidates.",
        "Full preprocessing pipeline: cyclical time features, chronological split, "
        "train-only scaling, sliding-window dataset construction.",
        "Generalized, batched QLSTM cell (4-gate VQC architecture) implemented in PyTorch + "
        "PennyLane, adapted from the SPP-QLSTM reference to arbitrary feature counts.",
        "Automated hyperparameter search (Optuna) over qubit count, circuit depth, hidden size, "
        "and learning rate.",
        "Final QLSTM training with early stopping, checkpointing, and evaluation "
        "(RMSE / MAE / MAPE in original units) on a held-out chronological test split.",
        "Reproducible pipeline (run_pipeline.py) producing all artifacts under results/.",
    ], styles))
    story.append(Paragraph("<b>Not yet implemented (future phases):</b>", styles["Bodyc"]))
    story.append(bullets([
        "Dynamical Hamiltonian encoding: embedding the input directly into the time-evolution "
        "generator rather than a fixed angle-embedding + rotation ansatz.",
        "Classical LSTM and other variational-quantum baselines for comparative benchmarking "
        "(this checkpoint trains the QLSTM in isolation, as scoped for this review).",
        "Training on the full ETTh1 series (all 17,420 hours) and/or the other two candidate "
        "datasets, once Hamiltonian-encoding compute cost is characterized.",
        "Larger-scale / multi-seed hyperparameter optimization and statistical significance "
        "testing of results.",
        "Noise-model / real-hardware evaluation (the reference repository includes a noisy "
        "QLSTM variant; not yet ported).",
    ], styles))

    story.append(Paragraph("9. Known Limitations of This Checkpoint", styles["H1c"]))
    story.append(bullets([
        "Trained on a 3,500-hour subsample (not the full 17,420-hour series) purely for "
        "quantum-simulator compute tractability on CPU; results should be read as a "
        "proof-of-functioning-pipeline rather than a final accuracy claim.",
        "Hyperparameter search used short (4-epoch) trials to keep search cost bounded; "
        "ranking may shift with longer per-trial training.",
        "All quantum circuits are simulated classically via PennyLane's default.qubit device; "
        "no hardware-noise or real-QPU results are included.",
        "Single random seed used for the reported run; no variance/confidence interval across "
        "seeds yet.",
    ], styles))

    story.append(Paragraph("10. Repository Structure", styles["H1c"]))
    story.append(Paragraph(
        "data/ETTh1.csv &mdash; raw dataset &nbsp;|&nbsp; "
        "src/preprocessing.py &mdash; loading, feature engineering, splitting, scaling, windowing "
        "&nbsp;|&nbsp; src/qlstm_model.py &mdash; QLSTM cell + regressor &nbsp;|&nbsp; "
        "src/train.py &mdash; train/eval loops &nbsp;|&nbsp; "
        "src/hyperparam_search.py &mdash; Optuna search &nbsp;|&nbsp; "
        "run_pipeline.py &mdash; end-to-end orchestration &nbsp;|&nbsp; "
        "results/ &mdash; metrics, plots, checkpoint, logs &nbsp;|&nbsp; "
        "report/ &mdash; this document.", styles["Small"]))

    doc.build(story)
    print("Report written to report/QLSTM_Status_Report.pdf")


if __name__ == "__main__":
    main()
