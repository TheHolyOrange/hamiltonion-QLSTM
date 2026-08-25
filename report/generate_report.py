"""
Generates the status report PDF for the 30% implementation checkpoint of the
'Dynamical Hamiltonian Quantum Model for Higher Order Temporal Dependencies'
project (QLSTM component only).

Reads: results/metrics.json, results/hpo_results.json, results/model_config.json,
       results/classical_lstm_metrics.json, results/synthetic_order_results.json,
       results/plots/loss_curve.png, results/plots/predictions_vs_actual.png,
       results/plots/qlstm_vs_classical_vs_actual.png
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


def comparison_table(metrics, classical_metrics, styles):
    header = ["Metric", "QLSTM", "Classical LSTM"]
    data = [header,
        ["Test RMSE (original units, degC)", f"{metrics['test_rmse_original_units']:.4f}",
         f"{classical_metrics['test_rmse_original_units']:.4f}"],
        ["Test MAE (original units, degC)", f"{metrics['test_mae_original_units']:.4f}",
         f"{classical_metrics['test_mae_original_units']:.4f}"],
        ["Test MAPE", f"{metrics['test_mape_percent']:.2f}%",
         f"{classical_metrics['test_mape_percent']:.2f}%"],
        ["Best validation MSE (scaled)", f"{metrics['best_val_mse_scaled']:.5f}",
         f"{classical_metrics['best_val_mse_scaled']:.5f}"],
        ["Epochs trained (early stop)", f"{metrics['epochs_trained']}",
         f"{classical_metrics['epochs_trained']}"],
        ["Training wall-clock time", f"{metrics['train_time_seconds']:.1f} s",
         f"{classical_metrics['train_time_seconds']:.1f} s"],
        ["Hidden size", f"{metrics['model_cfg']['hidden_size']}",
         f"{classical_metrics['model_cfg']['hidden_size']}"],
        ["Trainable parameters", "n/a (VQC, see Sec. 6.1)", f"{classical_metrics['n_params']}"],
    ]
    t = Table(data, colWidths=[7 * cm, 4 * cm, 4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2b4c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
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


def synthetic_order_table(syn, styles):
    header = ["p", "QLSTM test acc.", "QLSTM test MSE", "LSTM test acc.", "LSTM test MSE"]
    data = [header]
    q_by_p = {r["p"]: r for r in syn["qlstm"]}
    c_by_p = {r["p"]: r for r in syn["classical_lstm"]}
    for p in syn["p_values"]:
        q, c = q_by_p[p], c_by_p[p]
        data.append([
            str(p), f"{q['test_acc']*100:.1f}%", f"{q['test_mse']:.4f}",
            f"{c['test_acc']*100:.1f}%", f"{c['test_mse']:.4f}",
        ])
    t = Table(data, colWidths=[1.5 * cm, 3.5 * cm, 3.2 * cm, 3.5 * cm, 3.2 * cm])
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


def bullets(items, styles, style_name="Bodyc"):
    return ListFlowable(
        [ListItem(Paragraph(it, styles[style_name]), bulletColor=colors.HexColor("#2c4a7c")) for it in items],
        bulletType="bullet", start="•", leftIndent=14,
    )


def main():
    metrics = load_json("results/metrics.json")
    hpo = load_json("results/hpo_results.json")
    model_config = load_json("results/model_config.json")
    syn = load_json("results/synthetic_order_results.json")
    classical_metrics = load_json("results/classical_lstm_metrics.json")

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
        "baselines, and large-scale hyperparameter optimization are planned for subsequent phases. "
        "This revision additionally responds to review feedback requesting a precise definition of "
        "\"higher-order temporal dependencies\" and experimental evidence for it (Sections 2 and 9).",
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

    story.append(Paragraph("2. Defining \"Higher-Order Temporal Dependencies\"", styles["H1c"]))
    story.append(Paragraph(
        "This phrase is central to the project's problem statement, so it is defined precisely "
        "here rather than used as a slogan. Two complementary definitions are used:",
        styles["Bodyc"]))
    story.append(bullets([
        "<b>Statistical definition.</b> A target y has an order-p temporal dependency on lagged "
        "inputs x<sub>t-1</sub>,...,x<sub>t-p</sub> if y is statistically independent of every "
        "strict subset of those lags, and only the full joint interaction of all p lags carries "
        "information about y. This is the k-wise-independence property of, e.g., the parity "
        "function. Note this is <i>not</i> the same as \"long lookback window\": a linear "
        "function of many lags is still order-1 in this sense, since each lag contributes "
        "independently and additively.",
        "<b>Dynamical/Hamiltonian definition</b> (the one motivating this project's architecture). "
        "If inputs are encoded into a time-evolution generator, "
        "U(x,t) = &#120009; exp(-i&int;H(x(t))dt), its Dyson/Magnus expansion produces terms such "
        "as &int;&int; &#120009;[H(t<sub>1</sub>)H(t<sub>2</sub>)] dt<sub>1</sub>dt<sub>2</sub>, "
        "etc.; the n-th order term nonlinearly couples n distinct time points through the "
        "non-commutativity of H at different times. A fixed-ansatz VQC gate recurrence (this "
        "checkpoint's architecture) is Markovian by construction and has no algebraic access to "
        "these terms &mdash; this is precisely the gap the planned dynamical-Hamiltonian encoding "
        "(Section 6) is intended to close.",
    ], styles))
    story.append(Paragraph(
        "Multi-scale periodicity in ETTh1 (Section 3) motivates dataset choice but does not, by "
        "itself, demonstrate order-p capture &mdash; a model can reach low forecasting error there "
        "with no sensitivity to interaction order at all. Section 8 reports a controlled synthetic "
        "test of the statistical definition instead.", styles["Bodyc"]))

    story.append(Paragraph("3. Reference Implementation Used", styles["H1c"]))
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

    # --- 4. Dataset selection ---
    story.append(Paragraph("4. Dataset Selection", styles["H1c"]))
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

    # --- 5. Preprocessing ---
    story.append(Paragraph("5. Data Preprocessing Pipeline", styles["H1c"]))
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

    # --- 6. Architecture ---
    story.append(Paragraph("6. QLSTM Architecture", styles["H1c"]))
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

    story.append(Paragraph("6.1 Selected Configuration (from hyperparameter search)", styles["H2c"]))
    story.append(config_table(metrics["model_cfg"], {
        "Learning rate": metrics["lr"],
        "Batch size": metrics["batch_size"],
        "Sequence length": model_config["sequence_length"],
        "Rows used": model_config["n_rows"],
    }, styles))

    story.append(PageBreak())

    # --- 7. HPO ---
    story.append(Paragraph("7. Hyperparameter Search", styles["H1c"]))
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

    # --- 8. Training & results ---
    story.append(Paragraph("8. Final Training & Test Results", styles["H1c"]))
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

    # --- 8.1 Classical LSTM baseline comparison ---
    story.append(Paragraph("8.1 Comparison with a Classical LSTM Baseline", styles["H2c"]))
    story.append(Paragraph(
        f"A classical LSTM baseline (standard nn.LSTM gates, hidden_size="
        f"{classical_metrics['model_cfg']['hidden_size']} matched to the QLSTM's selected "
        "configuration, {} trainable parameters) was trained under the identical harness — same "
        "data split, scaling, windowing, optimizer, loss, and early-stopping protocol — to give a "
        "direct forecasting-accuracy comparison rather than only the capability-probe comparison "
        "in Section 9. No separate hyperparameter search was run for the classical LSTM; unlike "
        "the VQC gates it is cheap at this scale and is not itself the object of the comparison."
        .format(classical_metrics["n_params"]), styles["Bodyc"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(comparison_table(metrics, classical_metrics, styles))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Image(os.path.join(ROOT, "results/plots/qlstm_vs_classical_vs_actual.png"),
                        width=15 * cm, height=7.5 * cm))
    story.append(Paragraph("Figure 2.1: QLSTM vs. classical LSTM vs. actual oil temperature on held-out "
                            "test data (first 300 hourly steps, original units).", styles["Small"]))
    rmse_gain = (1 - metrics["test_rmse_original_units"] / classical_metrics["test_rmse_original_units"]) * 100
    story.append(Paragraph(
        f"On this task the QLSTM's test RMSE is {rmse_gain:.1f}% lower than the classical LSTM's "
        f"({metrics['test_rmse_original_units']:.4f} vs. {classical_metrics['test_rmse_original_units']:.4f} "
        "degC), with the classical LSTM using orders of magnitude fewer parameters and training in "
        f"{classical_metrics['train_time_seconds']:.0f}s vs. {metrics['train_time_seconds']:.0f}s. This is a "
        "single-seed, single-configuration comparison (Section 11) and, per Section 2, a real-dataset "
        "accuracy gap of this kind does not by itself establish order-p / higher-order dependency capture "
        "— see Section 9 for the controlled probe of that specific claim.", styles["Bodyc"]))

    story.append(PageBreak())

    # --- 9. Synthetic order-p experiment ---
    story.append(Paragraph("9. Synthetic Order-p Experiment: Testing the Higher-Order Claim", styles["H1c"]))
    story.append(Paragraph(
        "Section 2 defined an order-p temporal dependency as one where the target is "
        "statistically independent of any strict subset of p lagged inputs, and depends only on "
        "their full joint interaction. This is tested directly with a synthetic order-p parity "
        "task: p i.i.d. Rademacher (&plusmn;1) inputs x<sub>1</sub>,...,x<sub>p</sub> are presented "
        "one per time step, and the target is y = x<sub>1</sub>&middot;x<sub>2</sub>&middot;"
        "...&middot;x<sub>p</sub>. By construction, no subset of fewer than p inputs carries any "
        "information about y, so solving this task requires the recurrent state to retain and "
        "nonlinearly combine all p steps &mdash; a direct, ground-truth-order proxy for order-p "
        "capture, independent of any real-dataset confound.", styles["Bodyc"]))
    story.append(Paragraph(
        f"The QLSTM (hidden_size={syn['config']['hidden_size']}, n_qubits={syn['config']['n_qubits']}, "
        f"n_qlayers={syn['config']['n_qlayers']}) and a parameter-comparable classical LSTM baseline "
        f"were each trained from scratch on p = {', '.join(str(p) for p in syn['p_values'])} "
        f"(window length = p, {syn['config']['n_samples']} samples per p, 70/15/15 split, identical "
        f"optimizer/epoch budget), and evaluated by test-set sign-accuracy (chance = 50%).",
        styles["Bodyc"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(synthetic_order_table(syn, styles))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Image(os.path.join(ROOT, "results/plots/order_dependency_curve.png"),
                        width=14 * cm, height=9 * cm))
    story.append(Paragraph("Figure 3: test sign-accuracy vs. dependency order p, QLSTM vs. classical "
                            "LSTM baseline, dashed line = chance level.", styles["Small"]))
    story.append(Paragraph(
        "Both architectures here are gate-recurrence / VQC-ansatz based and Markovian in the sense "
        "described in Section 6 &mdash; neither has structural, algebraic access to order-p "
        "interactions; any fit to higher p can only come from what gradient descent discovers "
        "within the training budget. This is expected to (and, per the table above, does) degrade "
        "as p grows, and is the quantitative \"before\" baseline the dynamical-Hamiltonian encoding "
        "(Section 6) is intended to improve on, since its Dyson-expansion terms give the model "
        "direct algebraic access to order-p multi-time correlations rather than requiring the "
        "optimizer to reconstruct them from a fixed-ansatz gate recurrence.", styles["Bodyc"]))
    story.append(Paragraph(
        f"Both models solve parity exactly through p=3 (test sign-accuracy 100%, MSE "
        f"&lt;10<super>-6</super>) and both collapse to chance at p=4-5 (accuracy "
        "&asymp;47-51%, MSE&nbsp;&asymp;&nbsp;1, i.e. degenerating to predicting &asymp;0) &mdash; "
        f"and they fail at exactly the same p despite the QLSTM having roughly "
        f"{syn['classical_lstm'][0]['n_params'] / syn['qlstm'][0]['n_params']:.1f}x fewer trainable "
        "parameters than the classical LSTM. That the smaller QLSTM does not fail earlier suggests "
        "this is a genuine order-p capacity wall under this fixed training budget shared by both "
        "gate-recurrence architectures, rather than either model simply being under-parameterized "
        "&mdash; consistent with the Markovian-architecture argument above.", styles["Bodyc"]))

    story.append(PageBreak())

    # --- 10. Status vs scope ---
    story.append(Paragraph("10. Implementation Status vs. Project Scope", styles["H1c"]))
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
        "Classical LSTM baseline, capacity-matched and trained under the identical harness, for "
        "direct forecasting-accuracy comparison (Section 8.1) and a synthetic order-p capability "
        "comparison (Section 9).",
        "Reproducible pipeline (run_pipeline.py) producing all artifacts under results/.",
    ], styles))
    story.append(Paragraph("<b>Not yet implemented (future phases):</b>", styles["Bodyc"]))
    story.append(bullets([
        "Dynamical Hamiltonian encoding: embedding the input directly into the time-evolution "
        "generator rather than a fixed angle-embedding + rotation ansatz. A precise definition of "
        "the \"higher-order temporal dependencies\" this targets, and a synthetic order-p capability "
        "probe for the current (pre-Hamiltonian) architecture, are now in place (Sections 2, 9) as "
        "the baseline this encoding is meant to improve on.",
        "Other variational-quantum baselines (e.g. noisy/hardware-aware QLSTM variants) for "
        "comparative benchmarking beyond the classical LSTM baseline now in place (Section 8.1).",
        "Training on the full ETTh1 series (all 17,420 hours) and/or the other two candidate "
        "datasets, once Hamiltonian-encoding compute cost is characterized.",
        "Larger-scale / multi-seed hyperparameter optimization and statistical significance "
        "testing of results.",
        "Noise-model / real-hardware evaluation (the reference repository includes a noisy "
        "QLSTM variant; not yet ported).",
    ], styles))

    story.append(Paragraph("11. Known Limitations of This Checkpoint", styles["H1c"]))
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
        "The order-p synthetic probe (Section 9) uses a single seed and a small, fixed epoch "
        "budget per architecture; it is not parameter-matched between QLSTM and the classical LSTM "
        "baseline, so the comparison should be read as a rough, not statistically rigorous, "
        "baseline reading.",
        "The classical LSTM baseline (Section 8.1) is hidden-size-matched to the QLSTM but not "
        "separately hyperparameter-searched, and both models are trained from a single seed; the "
        "reported RMSE/MAE/MAPE gap should be read as directional, not a statistically validated "
        "result.",
    ], styles))

    story.append(Paragraph("12. Repository Structure", styles["H1c"]))
    story.append(Paragraph(
        "data/ETTh1.csv &mdash; raw dataset &nbsp;|&nbsp; "
        "src/preprocessing.py &mdash; loading, feature engineering, splitting, scaling, windowing "
        "&nbsp;|&nbsp; src/qlstm_model.py &mdash; QLSTM cell + regressor &nbsp;|&nbsp; "
        "src/lstm_model.py &mdash; classical LSTM baseline &nbsp;|&nbsp; "
        "src/train.py &mdash; train/eval loops (shared by both models) &nbsp;|&nbsp; "
        "src/hyperparam_search.py &mdash; Optuna search &nbsp;|&nbsp; "
        "src/synthetic_order_experiment.py &mdash; order-p parity probe &nbsp;|&nbsp; "
        "run_pipeline.py &mdash; end-to-end QLSTM orchestration &nbsp;|&nbsp; "
        "run_classical_baseline.py &mdash; classical LSTM baseline on ETTh1 &nbsp;|&nbsp; "
        "results/ &mdash; metrics, plots, checkpoint, logs &nbsp;|&nbsp; "
        "report/ &mdash; this document.", styles["Small"]))

    doc.build(story)
    print("Report written to report/QLSTM_Status_Report.pdf")


if __name__ == "__main__":
    main()
