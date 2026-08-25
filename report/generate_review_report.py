"""
Builds the Review Report I PDF for the H-QLSTM project, replicating the
structure/styling of the ssnreview LaTeX class (title page -> roman-numbered
front matter [Contents / List of Figures / List of Tables] -> arabic body
chapters -> references) using ReportLab, since no LaTeX engine is available
in this environment.

Two-pass build: pass 1 renders the body chapters alone to discover the exact
arabic page number each heading/figure/table caption lands on; pass 2 renders
the full document (title + front matter built from those recorded page
numbers + body + references).
"""
import io
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, NextPageTemplate, PageBreak,
    Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, ListFlowable, ListItem
)
from reportlab.platypus.flowables import HRFlowable

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIGS = os.path.join(HERE, "figures")
PLOTS = os.path.join(REPO, "results", "plots")

NAVY = colors.HexColor("#16305c")
LIGHTGREY = colors.HexColor("#666666")

LEFT_M, RIGHT_M, TOP_M, BOT_M = 1.25 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch
PAGEW, PAGEH = A4
FRAME_W = PAGEW - LEFT_M - RIGHT_M
FRAME_H = PAGEH - TOP_M - BOT_M

FRONT_PAGES = 4  # title, contents, list-of-figures, list-of-tables (each forced to exactly one page)

# ---------------------------------------------------------------- styles ---
S = {}
S["body"] = ParagraphStyle("body", fontName="Times-Roman", fontSize=11.5, leading=16.5,
                            alignment=TA_JUSTIFY, spaceAfter=8, firstLineIndent=0)
S["chapter"] = ParagraphStyle("Chapter", fontName="Times-Bold", fontSize=17, leading=20,
                               spaceBefore=4, spaceAfter=14, textColor=colors.black)
S["section"] = ParagraphStyle("Section", fontName="Times-Bold", fontSize=13.2, leading=16,
                               spaceBefore=10, spaceAfter=8, textColor=colors.black)
S["subsection"] = ParagraphStyle("Subsection", fontName="Times-Bold", fontSize=11.8, leading=14,
                                  spaceBefore=8, spaceAfter=6, textColor=colors.black)
S["figcap"] = ParagraphStyle("FigCaption", fontName="Times-Roman", fontSize=10.3, leading=13,
                              alignment=TA_CENTER, spaceBefore=4, spaceAfter=12, textColor=colors.black)
S["tabcap"] = ParagraphStyle("TabCaption", fontName="Times-Roman", fontSize=10.3, leading=13,
                              alignment=TA_CENTER, spaceBefore=2, spaceAfter=6, textColor=colors.black)
S["note"] = ParagraphStyle("Note", fontName="Times-Italic", fontSize=10.8, leading=15,
                            textColor=colors.HexColor("#b03030"), spaceBefore=4, spaceAfter=10,
                            alignment=TA_JUSTIFY)
S["kw"] = ParagraphStyle("Keywords", fontName="Times-Roman", fontSize=11.5, leading=15, spaceBefore=6)
S["tblhead"] = ParagraphStyle("TblHead", fontName="Times-Bold", fontSize=9.6, leading=12, textColor=colors.white)
S["tblcell"] = ParagraphStyle("TblCell", fontName="Times-Roman", fontSize=9.3, leading=12.2)
S["toc_c"] = ParagraphStyle("TOCc", fontName="Times-Bold", fontSize=11.3, leading=16, spaceBefore=4)
S["toc_s"] = ParagraphStyle("TOCs", fontName="Times-Roman", fontSize=10.6, leading=14.5, leftIndent=14)
S["toc_ss"] = ParagraphStyle("TOCss", fontName="Times-Roman", fontSize=10.2, leading=13.5, leftIndent=28)
S["toc_page"] = ParagraphStyle("TOCp", fontName="Times-Roman", fontSize=10.6, leading=14.5, alignment=TA_LEFT)
S["frontheading"] = ParagraphStyle("FrontHeading", fontName="Times-Bold", fontSize=16, leading=20,
                                    spaceAfter=18)
S["title_inst"] = ParagraphStyle("TInst", fontName="Times-Bold", fontSize=15.5, alignment=TA_CENTER, leading=19)
S["title_dept"] = ParagraphStyle("TDept", fontName="Times-Roman", fontSize=12.5, alignment=TA_CENTER, leading=16)
S["title_label"] = ParagraphStyle("TLabel", fontName="Times-Bold", fontSize=14, alignment=TA_CENTER, spaceAfter=10)
S["title_big"] = ParagraphStyle("TBig", fontName="Times-Bold", fontSize=19, alignment=TA_CENTER, leading=24)
S["title_review"] = ParagraphStyle("TReview", fontName="Times-Bold", fontSize=14.5, alignment=TA_CENTER,
                                    spaceBefore=16)
S["title_sub_label"] = ParagraphStyle("TSubLabel", fontName="Times-Bold", fontSize=12.5, alignment=TA_CENTER,
                                       spaceAfter=6)
S["title_names"] = ParagraphStyle("TNames", fontName="Times-Roman", fontSize=12.5, alignment=TA_CENTER, leading=19)
S["title_year"] = ParagraphStyle("TYear", fontName="Times-Roman", fontSize=12.5, alignment=TA_CENTER)
S["refs"] = ParagraphStyle("Refs", fontName="Times-Roman", fontSize=10.8, leading=15, spaceAfter=9,
                            leftIndent=16, firstLineIndent=-16)


def P(text, style="body"):
    return Paragraph(text, S[style])


# ------------------------------------------------------------- utilities ---
def to_roman(n):
    vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, sym in vals:
        while n >= v:
            out += sym
            n -= v
    return out


def styled_table(header, rows, col_widths, header_bg=NAVY, font_size=9.3, align_rows=None):
    data = [[Paragraph(h, S["tblhead"]) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), S["tblcell"]) for c in r])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("LINEBELOW", (0, 0), (-1, 0), 1.1, colors.black),
        ("LINEABOVE", (0, 0), (-1, 0), 1.3, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.3, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f2f5fa")))
    t.setStyle(TableStyle(style))
    return t


# ======================================================== BODY CONTENT =====
def build_body(plots_dir):
    """Returns the list of flowables making up chapters 1-6 (arabic-numbered
    body). Used identically in both the page-recording pass and the final
    build so page numbers line up exactly."""
    F = []

    # ---- Abstract (unnumbered chapter, page 1) ----
    F.append(P("Abstract", "chapter"))
    F.append(P(
        "Multivariate time-series forecasting in domains such as power-grid load management, "
        "transformer thermal monitoring, and climate modelling increasingly relies on recurrent "
        "architectures capable of capturing long-range and higher-order temporal dependencies. "
        "Classical LSTM networks, and recent hybrid Quantum LSTM (QLSTM) models, embed input data "
        "into variational quantum circuits (VQCs) through shallow per-timestep angle-embedding "
        "layers, which constrains their representational capacity and limits their ability to model "
        "continuous, coupled temporal dynamics. This project proposes a Hamiltonian-Encoded Quantum "
        "LSTM (H-QLSTM), in which the input at each time step parameterizes a learnable Hamiltonian "
        "operator H(x,&theta;), and the recurrent quantum state evolves under a Trotterized "
        "time-evolution operator U&nbsp;=&nbsp;exp(&minus;iH&Delta;t) derived from that Hamiltonian, "
        "rather than through static rotation gates. This dynamical-systems-based encoding is intended "
        "to more naturally capture coupled, higher-order temporal dependencies and long-range "
        "correlations in multivariate sequences than existing angle-embedding QLSTM designs.", "body"))
    F.append(P(
        "As a 30% checkpoint (Phase&nbsp;I) toward this goal, a classical LSTM baseline and an "
        "existing angle-embedding QLSTM baseline (a 4-gate variational quantum cell adapted from "
        "Kea&nbsp;et&nbsp;al., 2024) were implemented and trained on the Electricity Transformer "
        "Temperature (ETTh1) dataset. The QLSTM baseline, configured with 6&nbsp;qubits and "
        "2&nbsp;variational layers over 24-hour lookback windows, achieved a test RMSE of "
        "0.59&nbsp;&deg;C, MAE of 0.39&nbsp;&deg;C, and MAPE of 4.89% after an Optuna-based "
        "hyperparameter search and early-stopped training. Phase&nbsp;II will design, implement, and "
        "integrate the proposed Hamiltonian encoding into the QLSTM cell, extend evaluation to the "
        "Electricity Load Diagrams and Jena Climate datasets, and benchmark the H-QLSTM against both "
        "baselines. The expected contribution is a systematic framework for designing, optimizing, "
        "and benchmarking Hamiltonian-driven quantum recurrent models for multivariate time-series "
        "forecasting.", "body"))
    F.append(P("<b>Keywords:</b> Quantum Machine Learning; Quantum LSTM; Hamiltonian Encoding; "
                "Variational Quantum Circuits; Multivariate Time-Series Forecasting", "kw"))
    F.append(PageBreak())

    # ================================================== 1. Introduction ====
    F.append(P("1.&nbsp;&nbsp;Introduction", "chapter"))
    F.append(P(
        "Sequential, physically-grounded systems &mdash; power grids, electrical transformers, and "
        "regional climate &mdash; generate multivariate time series whose future values depend on "
        "nonlinear, multi-scale, and higher-order couplings between past observations. Forecasting "
        "such series accurately supports load balancing, preventive transformer maintenance, and "
        "climate-risk planning. Recurrent neural networks, and quantum-enhanced variants of them, are "
        "the dominant modelling family for this task; this project examines how the input encoding "
        "used inside a quantum recurrent cell affects its ability to represent those dependencies.",
        "body"))

    F.append(P("1.1&nbsp;&nbsp;Background", "section"))
    F.append(P(
        "Classical Long Short-Term Memory (LSTM) networks use gated memory cells &mdash; input, "
        "forget, cell, and output gates &mdash; to mitigate the vanishing-gradient problem and retain "
        "information over long sequences. Their representational capacity is nonetheless bounded by "
        "classical linear-algebraic gate operations. Quantum machine learning models sequence data "
        "using variational quantum circuits (VQCs): parameterized rotation gates interleaved with "
        "entangling layers, trained by gradient descent via the parameter-shift rule. Existing Quantum "
        "LSTM (QLSTM) architectures replace each of the four classical LSTM gates with a small VQC, "
        "embedding the input into per-timestep gate rotations through angle embedding (data values "
        "mapped directly to rotation angles). Hamiltonian embedding, demonstrated for static data by "
        "Wang&nbsp;et&nbsp;al. (2025), instead encodes data into the coefficients of a Hamiltonian "
        "operator H(x) and evolves the quantum state under the unitary U&nbsp;=&nbsp;exp(&minus;iH&Delta;t), "
        "giving continuous, physically-motivated dynamics in place of discrete gate rotations.", "body"))
    F.append(P(
        "<b>Defining &ldquo;higher-order temporal dependency&rdquo;.</b> The phrase is used precisely "
        "here rather than as a slogan. Statistically, a target y has an order-p dependency on lagged "
        "inputs x<sub>t&minus;1</sub>,&hellip;,x<sub>t&minus;p</sub> if y is independent of every "
        "strict subset of those lags and only their full joint interaction carries information about "
        "y &mdash; the k-wise-independence property of, e.g., the parity function; a linear function of "
        "many lags is still order-1 in this sense, so &ldquo;higher order&rdquo; is not the same as "
        "&ldquo;long lookback window&rdquo;. Dynamically, if inputs are encoded into a time-evolution "
        "generator U(x,t)&nbsp;=&nbsp;&Tau;exp(&minus;i&int;H(x(t))dt), its Dyson/Magnus expansion "
        "produces terms such as &int;&int;&Tau;[H(t<sub>1</sub>)H(t<sub>2</sub>)]dt<sub>1</sub>dt<sub>2</sub>, "
        "whose n-th order term nonlinearly couples n distinct time points through the non-commutativity "
        "of H at different times &mdash; a fixed-ansatz VQC gate recurrence (the Phase&nbsp;I baseline) "
        "is Markovian by construction and has no algebraic access to these terms, which is exactly the "
        "gap the proposed Hamiltonian encoding targets.", "body"))

    F.append(P("1.2&nbsp;&nbsp;Motivation", "section"))
    F.append(P(
        "A gap exists between the static, shallow angle-embedding used in current QLSTM designs and "
        "the continuous, coupled temporal evolution that many physical time series actually follow. "
        "Embedding the input into the generator of the circuit's own time evolution &mdash; rather "
        "than into a fixed set of rotation gates &mdash; couples the quantum state's evolution "
        "directly to the input dynamics at every time step. This motivates a Hamiltonian-encoded "
        "QLSTM cell that is expected to model higher-order dependencies and longer-range temporal "
        "correlations more effectively than existing angle-embedding QLSTM cells, particularly in "
        "domains such as transformer thermal behaviour and climate systems where the underlying "
        "process is itself a continuous dynamical system.", "body"))

    F.append(P("1.3&nbsp;&nbsp;Objectives", "section"))
    F.append(ListFlowable([
        ListItem(P("To implement and validate a classical LSTM and an existing angle-embedding "
                    "QLSTM as baseline models for multivariate time-series forecasting.", "body")),
        ListItem(P("To design a Hamiltonian-Encoded QLSTM (H-QLSTM) cell in which input features "
                    "parameterize a learnable Hamiltonian and the recurrent state evolves under its "
                    "time-evolution operator.", "body")),
        ListItem(P("To implement the proposed H-QLSTM using PennyLane and PyTorch and integrate it "
                    "into the existing preprocessing / training / evaluation pipeline.", "body")),
        ListItem(P("To evaluate all three models on the ETTh1, Electricity Load Diagrams, and Jena "
                    "Climate datasets using RMSE, MAE, and MAPE.", "body")),
        ListItem(P("To compare the proposed H-QLSTM against the classical LSTM and existing QLSTM "
                    "baselines, particularly with respect to longer lookback windows and higher-order "
                    "dependency capture.", "body")),
    ], bulletType="1", start="1", leftIndent=18, spaceBefore=2, spaceAfter=10))

    F.append(P("1.4&nbsp;&nbsp;Scope", "section"))
    F.append(P(
        "<b>Included:</b> design and implementation of a classical LSTM, an existing angle-embedding "
        "QLSTM, and the proposed H-QLSTM; simulated quantum circuit execution (PennyLane "
        "<i>default.qubit</i>) sized to remain classically tractable (approximately 6&ndash;8 qubits, "
        "2&ndash;3 variational layers); evaluation on three public multivariate time-series datasets "
        "(ETTh1, Electricity Load Diagrams, Jena Climate). <b>Excluded:</b> execution on physical "
        "quantum hardware; real-time or streaming deployment; forecasting horizons beyond the windowed "
        "single/multi-step setting already validated in the Phase&nbsp;I checkpoint.", "body"))
    F.append(PageBreak())

    # ================================================== 2. Related Work ====
    F.append(P("2.&nbsp;&nbsp;Related Work", "chapter"))
    F.append(P(
        "Table&nbsp;2.1 summarizes the principal work this project builds on and positions itself "
        "against: hybrid quantum-classical recurrent forecasting, Hamiltonian-based quantum data "
        "embedding, and quantum recurrent forecasting architectures more broadly.", "body"))

    rw_header = ["S.No.", "Author/Year", "Proposed Method", "Key Findings", "Limitations"]
    rw_rows = [
        ["1", "Kea et al.\n(2024) [1]",
         "4-gate QLSTM cell: each classical LSTM gate replaced by a VQC with angle-embedded "
         "features and entangling layers.",
         "Competitive with classical LSTM/GRU baselines on stock-price forecasting with fewer "
         "trainable parameters.",
         "Data embedded via static per-timestep angle rotations; no explicit modelling of continuous "
         "temporal dynamics; single-domain (stock) evaluation."],
        ["2", "Wang et al.\n(2025) [2]",
         "Quantum Hamiltonian embedding: data reuploaded as coefficients of a Hamiltonian, state "
         "evolved via its time-evolution operator.",
         "Richer expressivity and improved trainability (fewer barren-plateau issues) versus gate-angle "
         "embedding on image classification.",
         "Applied to static image data; no recurrent/sequential architecture or temporal-dependency "
         "modelling."],
        ["3", "Moon et al.\n(2025) [3]",
         "QSegRNN: a segment-wise quantum recurrent unit for time-series forecasting.",
         "Reduces circuit depth and parameter count versus standard quantum RNNs while improving "
         "forecasting accuracy on benchmark series.",
         "Segment gating still relies on variational angle embedding; limited capacity for long-range, "
         "higher-order dependency modelling."],
        ["4", "Hochreiter &amp;\nSchmidhuber (1997) [4]",
         "Classical LSTM: gated recurrent memory cell with input/forget/output gates.",
         "Mitigates vanishing gradients; remains the standard classical baseline for sequence "
         "modelling.",
         "Purely classical representational capacity; motivates hybrid quantum approaches for richer "
         "feature spaces."],
    ]
    tbl = styled_table(rw_header, rw_rows,
                        col_widths=[1.3 * cm, 2.6 * cm, 4.6 * cm, 4.6 * cm, 4.6 * cm])
    F.append(KeepTogether([tbl, P("Table 2.1: Summary of related work", "tabcap")]))
    F.append(P(
        "Summary of Related Works table is mandatory and it should list the mainly referred documents "
        "for your work. Don&rsquo;t change the table column headings.", "note"))

    F.append(P("2.1&nbsp;&nbsp;Research Gap", "section"))
    F.append(P(
        "Existing hybrid QLSTM and quantum-RNN designs (Kea&nbsp;et&nbsp;al.; Moon&nbsp;et&nbsp;al.) "
        "embed data through shallow, static variational angle encodings within the gate circuits, "
        "which does not explicitly model continuous-time dynamics. Hamiltonian embedding "
        "(Wang&nbsp;et&nbsp;al.) has been shown to improve expressivity for static data but has not "
        "been integrated into a recurrent, sequential architecture. No systematic framework currently "
        "exists to design, optimize, and benchmark a Hamiltonian-driven QLSTM against classical and "
        "variational-quantum baselines for multivariate time-series forecasting with higher-order and "
        "long-range temporal dependencies &mdash; this is the gap this project addresses.", "body"))
    F.append(PageBreak())

    # ============================================== 3. Problem statement ===
    F.append(P("3.&nbsp;&nbsp;Problem statement", "chapter"))
    F.append(P(
        "Classical and current hybrid quantum recurrent models struggle to efficiently capture "
        "higher-order temporal dependencies in complex nonlinear time series due to limited "
        "representational capacity and shallow data encodings. Quantum LSTM (QLSTM) frameworks have "
        "shown promise in exploiting quantum feature spaces for sequence learning, but existing "
        "designs primarily embed data via variational circuits without explicit Hamiltonian-driven "
        "dynamics to model continuous temporal evolution. A dynamical Hamiltonian encoding integrated "
        "within a QLSTM can embed data into the evolution operator itself, enabling richer modelling "
        "of temporal couplings and long-range dependencies. However, no systematic framework has been "
        "developed to design, optimize, and benchmark such Hamiltonian-based QLSTM architectures "
        "against classical and variational quantum baselines. Addressing this gap will advance quantum "
        "sequence learning and open pathways for scalable, dynamical-system-based temporal models.",
        "body"))
    F.append(P("<b>Input:</b> windowed multivariate time-series sequences (24-hour lookback windows) "
                "drawn from the ETTh1, Electricity Load Diagrams, and Jena Climate datasets, with "
                "cyclical time features and scaled numerical features.", "body"))
    F.append(P("<b>Output:</b> a forecast of the target variable (e.g. oil temperature for ETTh1) at "
                "the next timestep(s), together with RMSE, MAE, and MAPE against held-out ground truth.",
                "body"))
    F.append(P("<b>Constraints:</b> the quantum circuit must remain classically simulatable for "
                "benchmarking (bounded qubit count and circuit depth, PennyLane <i>default.qubit</i> "
                "on CPU); training must use standard gradient-based optimization (autodiff / "
                "parameter-shift); all three models (classical LSTM, existing QLSTM, proposed H-QLSTM) "
                "must be trained and evaluated under identical data splits, preprocessing, and metrics.",
                "body"))
    F.append(P("<b>Criteria:</b> success is measured on the real datasets by RMSE, MAE, and MAPE "
                "relative to the classical LSTM and existing QLSTM baselines, and &mdash; since low "
                "error on ETTh1 alone does not demonstrate order-p sensitivity &mdash; by sign-accuracy "
                "on a controlled order-p parity diagnostic (Section&nbsp;5.2.4) that is provably "
                "unsolvable below the true interaction order p, giving a direct, swept measure of "
                "higher-order dependency capture rather than an inferred one.", "body"))
    F.append(PageBreak())

    # ============================ 4. Justification for the problem ========
    F.append(P("4.&nbsp;&nbsp;Justification for the problem formulation", "chapter"))
    F.append(P(
        "This chapter justifies the technical need for a Hamiltonian-encoded QLSTM, the feasibility "
        "of building and evaluating one within the project timeline, and the contribution it is "
        "expected to make.", "body"))

    F.append(P("4.1&nbsp;&nbsp;Need for the Proposed Work", "section"))
    F.append(P(
        "Quantum sequence models remain nascent, and no systematic Hamiltonian-based QLSTM benchmark "
        "currently exists in the literature. As quantum hardware and simulators mature, establishing "
        "whether a dynamically-motivated encoding meaningfully improves temporal-dependency modelling "
        "&mdash; versus simply adding circuit depth &mdash; is an open and practically important "
        "question for the quantum machine learning community.", "body"))

    F.append(P("4.2&nbsp;&nbsp;Feasibility", "section"))
    F.append(P(
        "<b>Technical:</b> the PennyLane + PyTorch stack required is already operational; a classical "
        "LSTM and an existing angle-embedding QLSTM baseline have been implemented, trained, and "
        "evaluated on ETTh1 as the Phase&nbsp;I checkpoint (Section&nbsp;5.3), confirming the pipeline "
        "end-to-end. <b>Operational:</b> a CPU-tractable configuration (6&nbsp;qubits, 2&nbsp;layers, "
        "hidden size&nbsp;8) has already been identified via Optuna hyperparameter search and trains "
        "in under an hour on a 3,500-row subsample. <b>Economic:</b> all datasets are public and free; "
        "no proprietary licences or paid compute are required for circuit simulation at this scale. "
        "<b>Schedule:</b> the remaining work &mdash; designing and implementing the Hamiltonian "
        "encoding, integrating it into the QLSTM cell, and benchmarking across three datasets &mdash; "
        "is scoped into Phase&nbsp;II (Chapter&nbsp;6).", "body"))

    F.append(P("4.3&nbsp;&nbsp;Expected Contribution", "section"))
    F.append(P(
        "The anticipated contribution is a systematic framework for designing, optimizing, and "
        "benchmarking Hamiltonian-driven quantum recurrent models: (i) a Hamiltonian-Encoded QLSTM "
        "cell that embeds input into a learnable Hamiltonian's time-evolution operator; (ii) an "
        "open, reproducible benchmark of this cell against a classical LSTM and an existing "
        "angle-embedding QLSTM across three public multivariate time-series datasets; and "
        "(iii) a quantified assessment of whether Hamiltonian encoding improves higher-order and "
        "long-range temporal dependency modelling over existing quantum recurrent designs.", "body"))
    F.append(PageBreak())

    # ======================================================= 5. Methodology
    F.append(P("5.&nbsp;&nbsp;Methodology", "chapter"))
    F.append(P(
        "The complete workflow spans data preprocessing, three parallel recurrent-model "
        "implementations sharing a common training/evaluation harness, and comparative benchmarking, "
        "as summarized below and shown in Figure&nbsp;5.1.", "body"))

    F.append(P("5.1&nbsp;&nbsp;Architectural Design for Proposed System", "section"))
    F.append(P(
        "Raw multivariate series are cleaned, time-encoded, scaled, and windowed by the input layer. "
        "The processing layer hosts three interchangeable recurrent components trained and evaluated "
        "under an identical harness: the classical LSTM baseline, the existing angle-embedding QLSTM "
        "baseline, and the proposed H-QLSTM, whose cell encodes the input into a learnable Hamiltonian "
        "H(x,&theta;) and evolves the quantum state via the Trotterized operator "
        "U&nbsp;=&nbsp;exp(&minus;iH&Delta;t) in place of static rotation gates. The output layer "
        "applies a classical regression head to the recurrent state and reports RMSE, MAE, and MAPE.",
        "body"))
    img = Image(os.path.join(FIGS, "architecture.png"))
    iw, ih = img.wrap(0, 0)
    max_w = FRAME_W
    if iw > max_w:
        scale = max_w / iw
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
    F.append(KeepTogether([img, P("Figure 5.1: High-level architecture of the proposed system",
                                   "figcap")]))

    F.append(P("5.2&nbsp;&nbsp;Modules", "section"))
    F.append(P("5.2.1&nbsp;&nbsp;Module 1: Data/Input Management", "subsection"))
    F.append(P(
        "Loads ETTh1 (and, in Phase&nbsp;II, the Electricity Load Diagrams and Jena Climate datasets), "
        "derives cyclical time-of-day / day-of-week features, performs a chronological train/"
        "validation/test split, applies feature scaling, and constructs 24-hour lookback windows for "
        "sequence-to-one supervised learning (<code>src/preprocessing.py</code>).", "body"))
    F.append(P("5.2.2&nbsp;&nbsp;Module 2: Core Processing", "subsection"))
    F.append(P(
        "Implements the recurrent cells: the classical LSTM; the existing 4-gate angle-embedding "
        "QLSTM cell (<code>src/qlstm_model.py</code>); and, in Phase&nbsp;II, the proposed H-QLSTM "
        "cell in which each gate's VQC is replaced by a Hamiltonian-parameterized time-evolution "
        "block.", "body"))
    F.append(P("5.2.3&nbsp;&nbsp;Module 3: User Interface and Output", "subsection"))
    F.append(P(
        "<code>run_pipeline.py</code> orchestrates preprocessing, hyperparameter search, final "
        "training, and evaluation end-to-end, writing metrics, loss curves, and prediction-versus-"
        "actual plots to <code>results/</code> for reporting.", "body"))
    F.append(P("5.2.4&nbsp;&nbsp;Module 4: Testing and Evaluation", "subsection"))
    F.append(P(
        "<code>train.py</code> implements the train/validation loop with early stopping; "
        "<code>hyperparam_search.py</code> runs an Optuna search over qubit count, layer depth, "
        "hidden size, and learning rate; <code>utils.py</code> computes RMSE, MAE, and MAPE used to "
        "benchmark all three models under identical conditions. "
        "<code>synthetic_order_experiment.py</code> provides a controlled capability probe that is "
        "independent of any real dataset: it constructs order-p parity tasks (y&nbsp;=&nbsp;x<sub>1</sub>"
        "&times;&hellip;&times;x<sub>p</sub> over i.i.d. Rademacher inputs), which are provably "
        "unsolvable from any strict subset of the p lagged inputs, and sweeps p to report sign-accuracy "
        "against chance (50%) for the QLSTM and a parameter-comparable classical LSTM &mdash; a direct, "
        "swept test of order-p capture that ETTh1 accuracy alone cannot provide, to be run against the "
        "H-QLSTM cell in Phase&nbsp;II once it is implemented.", "body"))

    F.append(P("5.3&nbsp;&nbsp;Preliminary Results (Phase&nbsp;I Checkpoint)", "section"))
    F.append(P(
        "The existing angle-embedding QLSTM baseline was trained on a 3,500-hour subsample of ETTh1 "
        "with 24-hour lookback windows, 6&nbsp;qubits, and 2&nbsp;variational layers, following an "
        "8-trial Optuna hyperparameter search. A classical LSTM baseline (standard nn.LSTM gates, "
        "hidden size 8, matched to the QLSTM's selected configuration) was trained under the "
        "identical harness &mdash; same data split, scaling, windowing, optimizer, loss, and "
        "early-stopping protocol &mdash; for a direct comparison; it was not separately "
        "hyperparameter-searched, since at this scale it is cheap and is not itself the object of "
        "the comparison. Table&nbsp;5.1 reports test-set performance for both; Figures&nbsp;5.2 and "
        "5.3 show the QLSTM's training/validation loss curve and predicted-versus-actual oil "
        "temperature, and Figure&nbsp;5.4 overlays both models' forecasts against actual values.",
        "body"))

    res_header = ["Metric", "QLSTM", "Classical LSTM"]
    res_rows = [
        ["Test RMSE (original units, &deg;C)", "0.5919", "0.6316"],
        ["Test MAE (original units, &deg;C)", "0.3909", "0.4021"],
        ["Test MAPE", "4.89%", "5.15%"],
        ["Best validation MSE (scaled)", "0.03926", "0.05039"],
        ["Epochs trained (early stop)", "14 (patience 8)", "28 (patience 8)"],
        ["Training wall-clock time", "52.3 min", "1.6 s"],
        ["Trainable parameters", "n/a (VQC circuit)", "681"],
        ["Model configuration", "6 qubits, 2 variational layers, hidden size 8, "
                                 "batch size 32, lr 0.01, sequence length 24 h",
         "hidden size 8, batch size 32, lr 0.01, sequence length 24 h"],
    ]
    rtbl = styled_table(res_header, res_rows, col_widths=[5.4 * cm, 5.4 * cm, 5.8 * cm])
    F.append(KeepTogether([rtbl, P("Table 5.1: Phase I checkpoint &mdash; QLSTM vs. classical LSTM "
                                    "baseline test performance on ETTh1. QLSTM test RMSE is ~6.7% "
                                    "lower; this is a single-seed, single-configuration comparison, "
                                    "not a statistically validated result.", "tabcap")]))

    img2 = Image(os.path.join(plots_dir, "loss_curve.png"))
    iw2, ih2 = img2.wrap(0, 0)
    if iw2 > max_w * 0.75:
        scale = (max_w * 0.75) / iw2
        img2.drawWidth = iw2 * scale
        img2.drawHeight = ih2 * scale
    F.append(KeepTogether([img2, P("Figure 5.2: Training/validation loss curve &mdash; existing "
                                    "QLSTM baseline (Phase I checkpoint)", "figcap")]))

    img3 = Image(os.path.join(plots_dir, "predictions_vs_actual.png"))
    iw3, ih3 = img3.wrap(0, 0)
    if iw3 > max_w * 0.85:
        scale = (max_w * 0.85) / iw3
        img3.drawWidth = iw3 * scale
        img3.drawHeight = ih3 * scale
    F.append(KeepTogether([img3, P("Figure 5.3: Predicted vs. actual oil temperature &mdash; existing "
                                    "QLSTM baseline (Phase I checkpoint)", "figcap")]))

    img4 = Image(os.path.join(plots_dir, "qlstm_vs_classical_vs_actual.png"))
    iw4, ih4 = img4.wrap(0, 0)
    if iw4 > max_w * 0.85:
        scale = (max_w * 0.85) / iw4
        img4.drawWidth = iw4 * scale
        img4.drawHeight = ih4 * scale
    F.append(KeepTogether([img4, P("Figure 5.4: QLSTM vs. classical LSTM vs. actual oil temperature "
                                    "on held-out test data (Phase I checkpoint)", "figcap")]))
    F.append(PageBreak())

    # ============================================ 6. Project timeline chart
    F.append(P("6.&nbsp;&nbsp;Project timeline chart", "chapter"))
    F.append(P(
        "Figure&nbsp;6.1 shows the two-phase project schedule. Phase&nbsp;I (August&ndash;December "
        "2026) covers baseline development and the design of the proposed architecture; Phase&nbsp;II "
        "(January&ndash;June 2027) covers implementation, integration, optimization, and final "
        "benchmarking of the H-QLSTM.", "body"))
    img4 = Image(os.path.join(FIGS, "timeline_gantt.png"))
    iw4, ih4 = img4.wrap(0, 0)
    if iw4 > max_w:
        scale = max_w / iw4
        img4.drawWidth = iw4 * scale
        img4.drawHeight = ih4 * scale
    F.append(KeepTogether([img4, P("Figure 6.1: Project timeline", "figcap")]))

    return F


def build_references():
    F = [P("References", "chapter")]
    refs = [
        "K. Kea, D. Kim, C. Huot, T.-K. Kim, and Y. Han, \"A Hybrid Quantum-Classical Model for Stock "
        "Price Prediction Using Quantum-Enhanced Long Short-Term Memory\", <i>Entropy</i>, vol.&nbsp;26, "
        "no.&nbsp;11, p.&nbsp;954, 2024. DOI: https://doi.org/10.3390/e26110954",

        "P. Wang, C. R. Myers, L. C. L. Hollenberg, et al., \"Quantum Hamiltonian embedding of images "
        "for data reuploading classifiers\", <i>Quantum Machine Intelligence</i>, vol.&nbsp;7, p.&nbsp;35, "
        "2025. DOI: https://doi.org/10.1007/s42484-025-00247-7",

        "K. H. Moon, S. G. Jeong, and W. J. Hwang, \"QSegRNN: quantum segment recurrent neural network "
        "for time series forecasting\", <i>EPJ Quantum Technology</i>, vol.&nbsp;12, p.&nbsp;32, 2025. "
        "DOI: https://doi.org/10.1140/epjqt/s40507-025-00333-6",

        "S. Hochreiter and J. Schmidhuber, \"Long Short-Term Memory\", <i>Neural Computation</i>, "
        "vol.&nbsp;9, no.&nbsp;8, pp.&nbsp;1735&ndash;1780, 1997. DOI: https://doi.org/10.1162/neco.1997.9.8.1735",

        "H. Zhou et al., \"ETDataset: Electricity Transformer Temperature dataset\", GitHub repository, "
        "2021. [Online]. Available: https://github.com/zhouhaoyi/ETDataset",

        "A. Trindade, \"ElectricityLoadDiagrams20112014 Data Set\", UCI Machine Learning Repository, "
        "2015. [Online]. Available: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014",

        "Max Planck Institute for Biogeochemistry, \"Jena Climate Dataset (Weather Station Beutenberg)\", "
        "2026. [Online]. Available: https://www.bgc-jena.mpg.de/wetter. [Accessed: 25-Aug-2026].",
    ]
    for i, r in enumerate(refs, 1):
        F.append(P(f"[{i}]&nbsp; {r}", "refs"))
    return F


# ================================================================ PASS 1 ===
class RecorderDoc(BaseDocTemplate):
    def __init__(self, *a, **kw):
        BaseDocTemplate.__init__(self, *a, **kw)
        self.headings = []   # (level, text, page)
        self.figures = []    # (text, page)
        self.tables = []     # (text, page)

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            text = flowable.getPlainText()
            if style_name == "Chapter":
                self.headings.append((0, text, self.page))
            elif style_name == "Section":
                self.headings.append((1, text, self.page))
            elif style_name == "Subsection":
                self.headings.append((2, text, self.page))
            elif style_name == "FigCaption":
                self.figures.append((text, self.page))
            elif style_name == "TabCaption":
                self.tables.append((text, self.page))


def record_pages(plots_dir):
    frame = Frame(LEFT_M, BOT_M, FRAME_W, FRAME_H, id="body")
    buf = io.BytesIO()
    doc = RecorderDoc(buf, pagesize=A4,
                       leftMargin=LEFT_M, rightMargin=RIGHT_M, topMargin=TOP_M, bottomMargin=BOT_M)
    doc.addPageTemplates([PageTemplate(id="body", frames=[frame])])
    story = build_body(plots_dir) + build_references()
    doc.build(story)
    return doc.headings, doc.figures, doc.tables


# ================================================================ PASS 2 ===
def footer_front(canvas, doc):
    canvas.saveState()
    pn = canvas.getPageNumber()
    canvas.setFont("Times-Roman", 10)
    canvas.drawCentredString(PAGEW / 2, 0.6 * inch, to_roman(pn - 1))
    canvas.restoreState()


def footer_body(canvas, doc):
    canvas.saveState()
    pn = canvas.getPageNumber()
    canvas.setFont("Times-Roman", 10)
    canvas.drawCentredString(PAGEW / 2, 0.6 * inch, str(pn - FRONT_PAGES))
    canvas.restoreState()


def footer_blank(canvas, doc):
    pass


def build_title_flowables(meta):
    F = []
    F.append(Spacer(1, 0.3 * cm))
    logo_path = meta.get("logo_path")
    if logo_path and os.path.exists(logo_path):
        li = Image(logo_path)
        li.drawWidth = 3.0 * cm
        li.drawHeight = 3.0 * cm * (li.imageHeight / li.imageWidth)
        li.hAlign = "CENTER"
        F.append(li)
    else:
        box = Table([[" "]], colWidths=[3.2 * cm], rowHeights=[2.4 * cm])
        box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.black),
                                  ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                  ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        F.append(box)
    F.append(Spacer(1, 0.5 * cm))
    F.append(P(meta["institution"], "title_inst"))
    F.append(Spacer(1, 0.15 * cm))
    F.append(P(meta["department"], "title_dept"))
    F.append(Spacer(1, 1.7 * cm))
    F.append(P("Title", "title_label"))
    F.append(P(meta["title"], "title_big"))
    F.append(P(meta["review"], "title_review"))
    F.append(Spacer(1, 1.7 * cm))
    F.append(P("Submitted by", "title_sub_label"))
    F.append(P("<br/>".join(meta["students"]), "title_names"))
    F.append(Spacer(1, 0.9 * cm))
    F.append(P("Under the guidance of", "title_sub_label"))
    F.append(P(meta["supervisor"], "title_names"))
    F.append(Spacer(1, 1.7 * cm))
    F.append(P(f"Academic Year: {meta['academic_year']}", "title_year"))
    return F


def build_front_lists(headings, figures, tables):
    """Returns (contents_flowables, lof_flowables, lot_flowables) as three
    separate lists (no embedded PageBreak/template flowables) so the caller
    can place NextPageTemplate immediately before each PageBreak -- doing it
    any other way makes reportlab apply the template one page late."""
    contents, lof, lot = [], [], []

    # ---- Contents ----
    F = contents
    F.append(P("Contents", "frontheading"))
    rows = []
    style_by_level = {0: "toc_c", 1: "toc_s", 2: "toc_ss"}
    for level, text, page in headings:
        rows.append([Paragraph(text, S[style_by_level[level]]),
                     Paragraph(str(page), S["toc_page"])])
    t = Table(rows, colWidths=[FRAME_W - 1.4 * cm, 1.4 * cm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 1),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("ALIGN", (1, 0), (1, -1), "RIGHT")]))
    F.append(t)

    # ---- List of Figures ----
    F = lof
    F.append(P("List of Figures", "frontheading"))
    rows = [[Paragraph(text, S["toc_s"]), Paragraph(str(page), S["toc_page"])] for text, page in figures]
    t2 = Table(rows, colWidths=[FRAME_W - 1.4 * cm, 1.4 * cm])
    t2.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                             ("TOPPADDING", (0, 0), (-1, -1), 3),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                             ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                             ("ALIGN", (1, 0), (1, -1), "RIGHT")]))
    F.append(t2)

    # ---- List of Tables ----
    F = lot
    F.append(P("List of Tables", "frontheading"))
    rows = [[Paragraph(text, S["toc_s"]), Paragraph(str(page), S["toc_page"])] for text, page in tables]
    t3 = Table(rows, colWidths=[FRAME_W - 1.4 * cm, 1.4 * cm])
    t3.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                             ("TOPPADDING", (0, 0), (-1, -1), 3),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                             ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                             ("ALIGN", (1, 0), (1, -1), "RIGHT")]))
    F.append(t3)
    return contents, lof, lot


def build_pdf(out_path, plots_dir, meta):
    headings, figures, tables = record_pages(plots_dir)
    # Strip the leading numeric "N.&nbsp;&nbsp;" markup duplication issues: text already plain via getPlainText()

    front_frame = Frame(LEFT_M, BOT_M, FRAME_W, FRAME_H, id="front")
    body_frame = Frame(LEFT_M, BOT_M, FRAME_W, FRAME_H, id="bodyf")
    title_frame = Frame(LEFT_M, BOT_M, FRAME_W, FRAME_H, id="titlef")

    doc = BaseDocTemplate(out_path, pagesize=A4,
                           leftMargin=LEFT_M, rightMargin=RIGHT_M, topMargin=TOP_M, bottomMargin=BOT_M,
                           title=meta["title"], author=", ".join(s.split(" (")[0] for s in meta["students"]))
    doc.addPageTemplates([
        PageTemplate(id="title", frames=[title_frame], onPage=footer_blank),
        PageTemplate(id="front", frames=[front_frame], onPage=footer_front),
        PageTemplate(id="body", frames=[body_frame], onPage=footer_body),
    ])

    contents, lof, lot = build_front_lists(headings, figures, tables)

    # NextPageTemplate must be emitted immediately before the PageBreak whose
    # resulting new page should use it -- reportlab applies a pending
    # template starting at the *next* page break, so setting it any earlier
    # (e.g. right after a PageBreak that already fired) makes it land one
    # page late.
    story = []
    story.append(NextPageTemplate("title"))
    story += build_title_flowables(meta)

    story.append(NextPageTemplate("front"))
    story.append(PageBreak())
    story += contents
    story.append(PageBreak())          # still 'front' pending -> LOF page
    story += lof
    story.append(PageBreak())          # still 'front' pending -> LOT page
    story += lot

    story.append(NextPageTemplate("body"))
    story.append(PageBreak())
    story += build_body(plots_dir)
    story += build_references()

    doc.build(story)
    return out_path


if __name__ == "__main__":
    import sys
    plots_dir = sys.argv[1] if len(sys.argv) > 1 else PLOTS
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "Review_Report_I.pdf")
    logo_path = sys.argv[3] if len(sys.argv) > 3 else None

    meta = {
        "institution": "Sri Sivasubramaniya Nadar College of Engineering",
        "department": "Department of Computer Science and Engineering",
        "title": "Hamiltonian-Encoded Quantum LSTM for Learning Higher-Order Dependencies "
                 "in Multivariate Time-Series Forecasting",
        "review": "REVIEW REPORT &ndash; I",
        "students": [
            "Student Name 1 (Register No.)",
            "Student Name 2 (Register No.)",
            "Student Name 3 (Register No.)",
            "Student Name 4 (Register No.)",
        ],
        "supervisor": "Dr./Mr./Ms. Supervisor Name",
        "academic_year": "2026&ndash;2027",
        "logo_path": logo_path,
    }
    build_pdf(out_path, plots_dir, meta)
    print("Wrote", out_path)
