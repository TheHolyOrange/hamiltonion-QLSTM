"""
Builds the Review I presentation (.pptx) for the H-QLSTM project, replicating
the visual language of the UG_Project_Presentation_Template (navy title bars,
white body, blue arrow bullets, styled tables/callout boxes) with real
project content, using python-pptx (no PowerPoint/LaTeX/Beamer available in
this environment).
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
PLOTS = os.path.join(HERE, "..", "results", "plots")

NAVY = RGBColor(0x0F, 0x2A, 0x54)
NAVY2 = RGBColor(0x16, 0x3A, 0x6B)
BLUE = RGBColor(0x2E, 0x74, 0xC9)
LIGHTBLUE_BG = RGBColor(0xEA, 0xF1, 0xFB)
LIGHTBLUE_BG2 = RGBColor(0xDD, 0xE9, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x55, 0x55, 0x55)
RED = RGBColor(0xB0, 0x30, 0x30)
ROWALT = RGBColor(0xF2, 0xF5, 0xFA)

SW, SH = Inches(13.333), Inches(7.5)
SHORT_TITLE = "H-QLSTM Forecasting"
DEPT = "Department of Computer Science and Engineering"

FONT = "Calibri"


def set_bg(slide, color=WHITE):
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = color


def no_line(shape):
    shape.line.fill.background()


def solid(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color


def add_title_bar(slide, title, subtitle=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.35), Inches(0.28),
                                  Inches(12.63), Inches(1.05) if subtitle else Inches(0.85))
    bar.adjustments[0] = 0.10
    solid(bar, NAVY2)
    no_line(bar)
    bar.shadow.inherit = False
    tf = bar.text_frame
    tf.margin_left = Inches(0.25); tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.06); tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = title
    r.font.size = Pt(26); r.font.bold = False; r.font.color.rgb = WHITE; r.font.name = FONT
    if subtitle:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.LEFT
        r2 = p2.add_run(); r2.text = subtitle
        r2.font.size = Pt(15); r2.font.color.rgb = RGBColor(0xCE, 0xDC, 0xF2); r2.font.name = FONT
    return bar


def add_footer(slide, page_num):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(7.14), SW, Inches(0.36))
    solid(bar, NAVY)
    no_line(bar)
    bar.shadow.inherit = False
    tf = bar.text_frame
    tf.margin_left = Inches(0.25); tf.margin_top = 0; tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = SHORT_TITLE
    r.font.size = Pt(10.5); r.font.color.rgb = WHITE; r.font.name = FONT

    box = slide.shapes.add_textbox(Inches(4.0), Inches(7.14), Inches(6.2), Inches(0.36))
    tf2 = box.text_frame; tf2.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf2.margin_left = 0; tf2.margin_top = 0; tf2.margin_bottom = 0
    p2 = tf2.paragraphs[0]; p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run(); r2.text = DEPT
    r2.font.size = Pt(10.5); r2.font.color.rgb = WHITE; r2.font.name = FONT

    box2 = slide.shapes.add_textbox(Inches(12.35), Inches(7.14), Inches(0.8), Inches(0.36))
    tf3 = box2.text_frame; tf3.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf3.margin_left = 0; tf3.margin_top = 0; tf3.margin_bottom = 0
    p3 = tf3.paragraphs[0]; p3.alignment = PP_ALIGN.RIGHT
    r3 = p3.add_run(); r3.text = f"{page_num}/19"
    r3.font.size = Pt(10.5); r3.font.color.rgb = WHITE; r3.font.name = FONT


def new_slide(prs, title=None, subtitle=None, page_num=None, bg=WHITE):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, bg)
    if title:
        add_title_bar(slide, title, subtitle)
    if page_num:
        add_footer(slide, page_num)
    return slide


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    return box, tf


def add_bullets(slide, items, x, y, w, h, size=15, color=BLACK, bullet="▶", bcolor=BLUE,
                 space_after=10, bold_lead=False):
    box, tf = textbox(slide, x, y, w, h)
    first = True
    for item in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(space_after)
        p.level = 0
        r0 = p.add_run(); r0.text = bullet + "  "
        r0.font.size = Pt(size); r0.font.color.rgb = bcolor; r0.font.bold = True; r0.font.name = FONT
        if isinstance(item, tuple):
            lead, rest = item
            r1 = p.add_run(); r1.text = lead
            r1.font.size = Pt(size); r1.font.bold = True; r1.font.color.rgb = color; r1.font.name = FONT
            r2 = p.add_run(); r2.text = rest
            r2.font.size = Pt(size); r2.font.color.rgb = color; r2.font.name = FONT
        else:
            r1 = p.add_run(); r1.text = item
            r1.font.size = Pt(size); r1.font.color.rgb = color; r1.font.name = FONT
    return box


def add_numbered(slide, items, x, y, w, h, size=15, color=BLACK, space_after=12):
    box, tf = textbox(slide, x, y, w, h)
    first = True
    for i, item in enumerate(items, 1):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(space_after)
        r0 = p.add_run(); r0.text = f"{i}.  "
        r0.font.size = Pt(size); r0.font.bold = True; r0.font.color.rgb = NAVY2; r0.font.name = FONT
        r1 = p.add_run(); r1.text = item
        r1.font.size = Pt(size); r1.font.color.rgb = color; r1.font.name = FONT
    return box


def callout_box(slide, x, y, w, h, heading, body, heading_bg=NAVY2, body_bg=LIGHTBLUE_BG,
                 heading_size=14, body_size=13):
    hb = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, Inches(0.42))
    hb.adjustments[0] = 0.18
    solid(hb, heading_bg); no_line(hb); hb.shadow.inherit = False
    tf = hb.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.15)
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = heading
    r.font.size = Pt(heading_size); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = FONT

    bb = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y + Inches(0.40),
                                 w, h - Inches(0.40))
    solid(bb, body_bg); no_line(bb); bb.shadow.inherit = False
    bb.adjustments[0] = 0.04
    tf2 = bb.text_frame; tf2.vertical_anchor = MSO_ANCHOR.TOP
    tf2.margin_left = Inches(0.18); tf2.margin_right = Inches(0.15)
    tf2.margin_top = Inches(0.12); tf2.word_wrap = True
    if isinstance(body, list):
        first = True
        for item in body:
            p = tf2.paragraphs[0] if first else tf2.add_paragraph()
            first = False
            p.alignment = PP_ALIGN.LEFT
            p.space_after = Pt(6)
            r0 = p.add_run(); r0.text = "▶  "
            r0.font.size = Pt(body_size); r0.font.bold = True; r0.font.color.rgb = BLUE; r0.font.name = FONT
            r1 = p.add_run(); r1.text = item
            r1.font.size = Pt(body_size); r1.font.color.rgb = BLACK; r1.font.name = FONT
    else:
        p = tf2.paragraphs[0]
        r = p.add_run(); r.text = body
        r.font.size = Pt(body_size); r.font.color.rgb = BLACK; r.font.name = FONT
    return hb, bb


def add_table(slide, x, y, w, h, header, rows, col_widths=None, header_size=12, cell_size=11):
    n_rows, n_cols = len(rows) + 1, len(header)
    gtable = slide.shapes.add_table(n_rows, n_cols, x, y, w, h)
    table = gtable.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            table.columns[i].width = cw
    for j, htext in enumerate(header):
        cell = table.cell(0, j)
        cell.text = htext
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY2
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        for r in p.runs:
            r.font.bold = True; r.font.size = Pt(header_size); r.font.color.rgb = WHITE; r.font.name = FONT
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.08)
        cell.margin_top = Inches(0.04); cell.margin_bottom = Inches(0.04)
    for i, row in enumerate(rows, 1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = ROWALT if i % 2 == 0 else WHITE
            p = cell.text_frame.paragraphs[0]
            for r in p.runs:
                r.font.size = Pt(cell_size); r.font.color.rgb = BLACK; r.font.name = FONT
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.03); cell.margin_bottom = Inches(0.03)
    # slim row height
    for r in table.rows:
        r.height = Emu(int(h / n_rows))
    return gtable


def add_picture_fit(slide, path, x, y, max_w, max_h, align_center=True):
    from PIL import Image
    im = Image.open(path)
    iw, ih = im.size
    ar = iw / ih
    w, h = max_w, max_w / ar
    if h > max_h:
        h = max_h
        w = max_h * ar
    px = x + (max_w - w) / 2 if align_center else x
    py = y
    return slide.shapes.add_picture(path, int(px), int(py), int(w), int(h))


# ============================================================== BUILD =====
def build(out_path, meta):
    prs = Presentation()
    prs.slide_width = SW
    prs.slide_height = SH

    # ---- Slide 1: Title ----
    s = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s, WHITE)
    bar = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(1.35), Inches(12.13), Inches(1.55))
    bar.adjustments[0] = 0.10
    solid(bar, NAVY2); no_line(bar); bar.shadow.inherit = False
    tf = bar.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
    tf.margin_left = Inches(0.3); tf.margin_right = Inches(0.3)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = meta["title"]
    r.font.size = Pt(25); r.font.color.rgb = WHITE; r.font.name = FONT
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run(); r2.text = meta["review"]
    r2.font.size = Pt(16); r2.font.color.rgb = RGBColor(0xCE, 0xDC, 0xF2); r2.font.name = FONT

    box, tf = textbox(s, Inches(1.0), Inches(3.3), Inches(11.3), Inches(1.4), MSO_ANCHOR.TOP)
    first = True
    for st in meta["students_ppt"]:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = st
        r.font.size = Pt(15); r.font.color.rgb = BLACK; r.font.name = FONT

    box, tf = textbox(s, Inches(1.0), Inches(4.85), Inches(11.3), Inches(0.5))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = f"Guided by: {meta['supervisor']}"
    r.font.size = Pt(13.5); r.font.color.rgb = GREY; r.font.name = FONT

    box, tf = textbox(s, Inches(1.0), Inches(5.35), Inches(11.3), Inches(0.9))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = meta["department"]
    r.font.size = Pt(13); r.font.color.rgb = GREY; r.font.name = FONT
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run(); r2.text = meta["institution"]
    r2.font.size = Pt(13); r2.font.color.rgb = GREY; r2.font.name = FONT

    box, tf = textbox(s, Inches(1.0), Inches(6.5), Inches(11.3), Inches(0.5))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = f"Academic Year {meta['academic_year']}"
    r.font.size = Pt(13); r.font.color.rgb = GREY; r.font.name = FONT

    # ---- Slide 2: Presentation Outline ----
    s = new_slide(prs, "Presentation Outline", page_num=2)
    items = ["Introduction", "Literature Review", "System Design", "Project Management"]
    for i in range(4):
        oval = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(1.25), Inches(1.95 + i * 0.98), Inches(0.42), Inches(0.42))
        solid(oval, NAVY2); no_line(oval); oval.shadow.inherit = False
        otf = oval.text_frame; otf.vertical_anchor = MSO_ANCHOR.MIDDLE
        op = otf.paragraphs[0]; op.alignment = PP_ALIGN.CENTER
        orun = op.add_run(); orun.text = str(i + 1)
        orun.font.size = Pt(15); orun.font.bold = True; orun.font.color.rgb = WHITE; orun.font.name = FONT
        tbox, ttf = textbox(s, Inches(1.9), Inches(1.95 + i * 0.98), Inches(8), Inches(0.5), MSO_ANCHOR.MIDDLE)
        tp = ttf.paragraphs[0]
        tr = tp.add_run(); tr.text = items[i]
        tr.font.size = Pt(19); tr.font.color.rgb = NAVY2; tr.font.name = FONT; tr.font.bold = True

    # ---- Slide 3: Background and Motivation ----
    s = new_slide(prs, "Background and Motivation", page_num=3)
    add_bullets(s, [
        "Multivariate time series from power grids, transformers, and climate systems exhibit "
        "nonlinear, multi-scale, higher-order temporal dependencies.",
        "“Higher order” is precise, not a slogan: an order-p dependency needs the joint "
        "interaction of p lags — no strict subset carries information (the k-wise-independence "
        "property of parity). A long lookback of independent linear terms is still order-1.",
        "Classical LSTM and existing angle-embedding QLSTM models embed data through shallow, "
        "static gate rotations, limiting representational capacity.",
        "Hamiltonian embedding has improved expressivity for static (image) data, but has not been "
        "used inside a recurrent cell for sequential/temporal data.",
    ], Inches(0.6), Inches(1.7), Inches(7.6), Inches(5), size=14.5)
    callout_box(s, Inches(8.55), Inches(1.75), Inches(4.15), Inches(2.3), "Core Motivation",
                "A dynamical, Hamiltonian-driven encoding may capture coupled temporal dependencies "
                "that static angle-embedding QLSTM cells cannot.")

    # ---- Slide 4: Problem Statement ----
    s = new_slide(prs, "Problem Statement", page_num=4)
    callout_box(s, Inches(0.6), Inches(1.55), Inches(12.13), Inches(1.55), "Problem",
                "Classical and hybrid quantum recurrent models struggle to capture higher-order "
                "temporal dependencies due to limited representational capacity and shallow "
                "(angle-embedding) data encodings; no systematic framework exists to design, "
                "optimize, and benchmark a Hamiltonian-driven QLSTM against classical and "
                "variational-quantum baselines.", body_size=13.5)
    add_bullets(s, [
        "What is the exact technical problem? Shallow VQC data encoding limits temporal-dependency "
        "capacity in QLSTM cells.",
        "What causes it? Angle embedding maps inputs to static rotation gates, not continuous dynamics.",
        "Why are existing approaches insufficient? No Hamiltonian-encoded QLSTM has been designed, "
        "implemented, or benchmarked for time-series forecasting.",
        "What constraints must the solution satisfy? Classically simulatable circuits; identical "
        "data/metrics across classical LSTM, existing QLSTM, and proposed H-QLSTM.",
    ], Inches(0.6), Inches(3.35), Inches(12.1), Inches(3.4), size=15)

    # ---- Slide 5: Project Objectives ----
    s = new_slide(prs, "Project Objectives", page_num=5)
    add_numbered(s, [
        "To implement and validate a classical LSTM and an existing angle-embedding QLSTM as "
        "baseline models for multivariate time-series forecasting.",
        "To design a Hamiltonian-Encoded QLSTM (H-QLSTM) cell in which input features parameterize "
        "a learnable Hamiltonian, with the recurrent state evolving under its time-evolution operator.",
        "To implement the proposed H-QLSTM using PennyLane and PyTorch, integrated into the existing "
        "preprocessing / training / evaluation pipeline.",
        "To evaluate all three models on the ETTh1, Electricity Load Diagrams, and Jena Climate "
        "datasets using RMSE, MAE, and MAPE.",
        "To compare the proposed H-QLSTM against both baselines, particularly for longer lookback "
        "windows and higher-order dependency capture.",
    ], Inches(0.8), Inches(1.85), Inches(11.6), Inches(5), size=16, space_after=16)

    # ---- Slide 6: Scope and Expected Outcomes ----
    s = new_slide(prs, "Scope and Expected Outcomes", page_num=6)
    callout_box(s, Inches(0.6), Inches(1.7), Inches(5.9), Inches(4.6), "Included", [
        "Classical LSTM, existing QLSTM, and proposed H-QLSTM implementations",
        "Simulated quantum circuits (PennyLane default.qubit, ~6-8 qubits)",
        "ETTh1, Electricity Load Diagrams, Jena Climate datasets",
        "Evaluation via RMSE, MAE, MAPE across lookback horizons",
    ], body_size=14)
    callout_box(s, Inches(6.8), Inches(1.7), Inches(5.9), Inches(4.6), "Excluded / Constraints", [
        "Execution on physical quantum hardware",
        "Real-time or streaming deployment",
        "Forecast horizons beyond the validated windowed single/multi-step setting",
        "Circuit depth/qubit count bounded by classical simulation cost",
    ], body_size=14)

    # ---- Slide 7: Literature Review ----
    s = new_slide(prs, "Literature Review", page_num=7)
    header = ["Ref.", "Method / Technology", "Key Contribution", "Limitation / Research Gap"]
    rows = [
        ["[1]", "Kea et al. (2024) - 4-gate angle-embedding QLSTM",
         "Competitive stock-price forecasting vs. classical LSTM/GRU",
         "Static angle embedding; no continuous temporal dynamics"],
        ["[2]", "Wang et al. (2025) - Quantum Hamiltonian embedding",
         "Richer expressivity, better trainability on image data",
         "Static data only; no recurrent/sequential architecture"],
        ["[3]", "Moon et al. (2025) - QSegRNN",
         "Fewer parameters, improved quantum-RNN forecasting",
         "Still angle-embedding based; limited long-range capacity"],
        ["[4]", "Hochreiter & Schmidhuber (1997) - Classical LSTM",
         "Standard gated recurrent baseline, mitigates vanishing gradients",
         "Purely classical representational ceiling"],
    ]
    add_table(s, Inches(0.5), Inches(1.7), Inches(12.3), Inches(4.4), header, rows,
              col_widths=[Inches(0.8), Inches(3.6), Inches(3.9), Inches(4.0)], header_size=13, cell_size=12)

    # ---- Slide 8: Research Gap and Proposed Contribution ----
    s = new_slide(prs, "Research Gap and Proposed Contribution", page_num=8)
    callout_box(s, Inches(0.6), Inches(1.7), Inches(5.9), Inches(4.6), "Identified Gap", [
        "Existing QLSTM/quantum-RNN designs use shallow, static angle embedding",
        "Hamiltonian embedding demonstrated only for static (image) data",
        "No systematic Hamiltonian-QLSTM design/benchmark framework exists",
    ], body_size=14.5)
    callout_box(s, Inches(6.8), Inches(1.7), Inches(5.9), Inches(4.6), "Proposed Contribution", [
        "Hamiltonian-Encoded QLSTM cell: input parameterizes a learnable Hamiltonian",
        "State evolves via Trotterized U = exp(-iHΔt) in place of static rotation gates",
        "Open benchmark vs. classical LSTM and existing QLSTM across 3 public datasets",
    ], body_size=14.5)

    # ---- Slide 9: Existing vs Proposed ----
    s = new_slide(prs, "Existing System vs. Proposed System", page_num=9)
    header = ["Criterion", "Existing (Angle-Embedding) QLSTM", "Proposed H-QLSTM"]
    rows = [
        ["Data encoding", "Static per-timestep angle rotations", "Learnable Hamiltonian H(x,θ)"],
        ["State evolution", "Discrete variational gate sequence", "Trotterized U = exp(-iHΔt)"],
        ["Temporal dynamics", "Not explicitly modelled", "Continuous, physically motivated"],
        ["Higher-order dependencies", "Limited by shallow encoding", "Expected improvement (to be benchmarked)"],
        ["Evaluated datasets (this project)", "ETTh1 (Phase I checkpoint)", "ETTh1, Electricity Load, Jena Climate"],
    ]
    add_table(s, Inches(0.5), Inches(1.7), Inches(12.3), Inches(4.6), header, rows,
              col_widths=[Inches(3.1), Inches(4.6), Inches(4.6)], header_size=13, cell_size=12.5)

    # ---- Slide 10: Proposed System Architecture ----
    s = new_slide(prs, "Proposed System Architecture", page_num=10)
    add_picture_fit(s, os.path.join(FIGS, "architecture.png"), Inches(0.4), Inches(1.7),
                     Inches(12.5), Inches(4.6))
    box, tf = textbox(s, Inches(0.8), Inches(6.35), Inches(11.7), Inches(0.6))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = ("Three interchangeable recurrent components share one preprocessing pipeline and "
              "evaluation harness for direct comparison.")
    r.font.size = Pt(13); r.font.italic = True; r.font.color.rgb = GREY; r.font.name = FONT

    # ---- Slide 11: System Modules ----
    s = new_slide(prs, "System Modules", page_num=11)
    add_numbered(s, [
        "Data/Input Management: loads ETTh1 / Electricity Load / Jena Climate, cyclical time "
        "features, chronological split.",
        "Preprocessing: scaling and 24-hour lookback windowing for sequence-to-one learning.",
        "Core Processing: classical LSTM, existing angle-embedding QLSTM, and proposed H-QLSTM cells.",
        "Storage / Pipeline: run_pipeline.py orchestrates preprocessing, HPO, training, evaluation.",
        "Evaluation: train.py (early stopping), hyperparam_search.py (Optuna), utils.py (RMSE/MAE/MAPE).",
    ], Inches(0.8), Inches(1.85), Inches(11.6), Inches(5), size=16, space_after=16)

    # ---- Slide 12: Methodology / Workflow ----
    s = new_slide(prs, "Methodology / Workflow", page_num=12)
    add_numbered(s, [
        "Requirement analysis and dataset selection (ETTh1, Electricity Load, Jena Climate)",
        "Data preprocessing and windowing",
        "Classical LSTM and existing QLSTM baseline development",
        "H-QLSTM design and implementation (Hamiltonian encoding block)",
        "Integration, training, and hyperparameter optimization",
        "Comparative evaluation and reporting",
    ], Inches(0.6), Inches(1.85), Inches(7.5), Inches(5), size=15, space_after=14)
    callout_box(s, Inches(8.4), Inches(1.85), Inches(4.3), Inches(3.4), "Evaluation Metrics", [
        "RMSE / MAE / MAPE on real datasets", "Order-p parity diagnostic (sign-accuracy vs. chance)",
        "Training wall-clock time", "Parameter / circuit-depth efficiency",
    ], body_size=13.5)

    # ---- Slide 13: Core Algorithm ----
    s = new_slide(prs, "Core Algorithm: Hamiltonian-Encoded QLSTM Cell", page_num=13)
    callout_box(s, Inches(0.6), Inches(1.7), Inches(12.13), Inches(4.6),
                "Algorithm: H-QLSTM gate evaluation", [
        "Receive input x_t and previous hidden state h_{t-1}; concatenate as v_t.",
        "Map v_t to Hamiltonian coefficients: H(v_t, θ) = Σ_k c_k(v_t,θ) P_k "
        "(Pauli-string basis).",
        "Evolve the quantum state via the Trotterized unitary U = exp(-iH(v_t,θ)Δt), "
        "applied per QLSTM gate (input / forget / cell / output).",
        "Measure expectation values and map to classical gate activations via a linear readout.",
        "Update cell and hidden state as in a standard LSTM; pass to the regression head for the "
        "forecast.",
    ], body_size=14.5, heading_size=15)

    # ---- Slide 14: Technology Stack ----
    s = new_slide(prs, "Technology Stack and Resources", page_num=14)
    header = ["Category", "Selected Technology / Resource"]
    rows = [
        ["Programming Language", "Python 3.12"],
        ["Frameworks / Libraries", "PyTorch, PennyLane, Optuna, NumPy, pandas"],
        ["Datasets", "ETTh1 (ETDataset); Electricity Load Diagrams (UCI); Jena Climate (MPI-BGC)"],
        ["Quantum simulation", "PennyLane default.qubit, 6-8 qubits, 2-3 variational layers"],
        ["Development Tools", "Git, matplotlib for plots, JSON-based results tracking"],
        ["Hardware", "CPU (circuit simulation tractable at this scale); no GPU/QPU required"],
    ]
    add_table(s, Inches(0.6), Inches(1.75), Inches(12.13), Inches(4.5), header, rows,
              col_widths=[Inches(3.4), Inches(8.73)], header_size=14, cell_size=13)

    # ---- Slide 15: Preliminary Results (Phase I Checkpoint) ----
    s = new_slide(prs, "Preliminary Results (Phase I Checkpoint)", page_num=15)
    header = ["Metric", "QLSTM", "Classical LSTM"]
    rows = [
        ["Test RMSE (°C)", "0.5919", "0.6316"],
        ["Test MAE (°C)", "0.3909", "0.4021"],
        ["Test MAPE", "4.89%", "5.15%"],
        ["Trainable parameters", "n/a (VQC circuit)", "681"],
        ["Training wall-clock time", "52.3 min", "1.6 s"],
    ]
    add_table(s, Inches(0.6), Inches(1.7), Inches(6.0), Inches(3.4), header, rows,
              col_widths=[Inches(2.3), Inches(1.85), Inches(1.85)], header_size=13, cell_size=12)
    add_picture_fit(s, os.path.join(PLOTS, "qlstm_vs_classical_vs_actual.png"),
                     Inches(6.85), Inches(1.7), Inches(5.9), Inches(3.5))
    box, tf = textbox(s, Inches(0.6), Inches(5.35), Inches(12.13), Inches(1.5))
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = ("Both models trained on ETTh1 under an identical harness (same split, scaling, "
              "windowing, optimizer, early stopping); classical LSTM hidden size matched to the "
              "QLSTM's Optuna-selected configuration (6 qubits, 2 layers, hidden size 8) rather than "
              "separately tuned. QLSTM test RMSE is ~6.7% lower, at far higher parameter/compute cost "
              "— a single-seed, single-configuration result, not yet statistically validated.")
    r.font.size = Pt(13); r.font.italic = True; r.font.color.rgb = GREY; r.font.name = FONT

    # ---- Slide 16: Project Timeline ----
    s = new_slide(prs, "Project Timeline", page_num=16)
    add_picture_fit(s, os.path.join(FIGS, "timeline_gantt.png"), Inches(0.3), Inches(1.65),
                     Inches(12.7), Inches(5.1))

    # ---- Slide 17: Team Responsibilities ----
    s = new_slide(prs, "Team Responsibilities", page_num=17)
    header = ["Team Member", "Responsibilities", "Current Status"]
    rows = [
        ["Student 1", "Literature review, problem formulation, report/documentation", "In progress"],
        ["Student 2", "Data preprocessing, classical LSTM & QLSTM baseline implementation", "Completed (Phase I)"],
        ["Student 3", "Hyperparameter search, training pipeline, evaluation metrics", "Completed (Phase I)"],
        ["Student 4", "H-QLSTM architecture design, Phase II integration planning", "In progress"],
    ]
    add_table(s, Inches(0.6), Inches(1.85), Inches(12.13), Inches(3.6), header, rows,
              col_widths=[Inches(2.3), Inches(7.83), Inches(2.0)], header_size=14, cell_size=13)

    # ---- Slide 18: References ----
    s = new_slide(prs, "References", page_num=18)
    refs = [
        "K. Kea, D. Kim, C. Huot, T.-K. Kim, and Y. Han, \"A Hybrid Quantum-Classical Model for Stock "
        "Price Prediction Using Quantum-Enhanced Long Short-Term Memory,\" Entropy, vol. 26, no. 11, "
        "p. 954, 2024.",
        "P. Wang, C. R. Myers, L. C. L. Hollenberg, et al., \"Quantum Hamiltonian embedding of images "
        "for data reuploading classifiers,\" Quantum Machine Intelligence, vol. 7, p. 35, 2025.",
        "K. H. Moon, S. G. Jeong, and W. J. Hwang, \"QSegRNN: quantum segment recurrent neural network "
        "for time series forecasting,\" EPJ Quantum Technology, vol. 12, p. 32, 2025.",
        "S. Hochreiter and J. Schmidhuber, \"Long Short-Term Memory,\" Neural Computation, vol. 9, "
        "no. 8, pp. 1735-1780, 1997.",
        "H. Zhou et al., \"ETDataset: Electricity Transformer Temperature dataset,\" GitHub, 2021. "
        "[Online]. Available: github.com/zhouhaoyi/ETDataset",
        "A. Trindade, \"ElectricityLoadDiagrams20112014 Data Set,\" UCI ML Repository, 2015.",
        "Max Planck Institute for Biogeochemistry, \"Jena Climate Dataset,\" 2026. [Online]. Available: "
        "bgc-jena.mpg.de/wetter",
    ]
    box, tf = textbox(s, Inches(0.7), Inches(1.75), Inches(11.9), Inches(5))
    first = True
    for i, ref in enumerate(refs, 1):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(10)
        r0 = p.add_run(); r0.text = f"[{i}]  "
        r0.font.size = Pt(13); r0.font.bold = True; r0.font.color.rgb = NAVY2; r0.font.name = FONT
        r1 = p.add_run(); r1.text = ref
        r1.font.size = Pt(13); r1.font.color.rgb = BLACK; r1.font.name = FONT

    # ---- Slide 18: Thank You ----
    s = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s, WHITE)
    box, tf = textbox(s, Inches(1), Inches(2.9), Inches(11.3), Inches(1.2))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = "Thank You"
    r.font.size = Pt(44); r.font.color.rgb = NAVY2; r.font.name = FONT
    box, tf = textbox(s, Inches(1), Inches(4.1), Inches(11.3), Inches(0.7))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = "Questions and Discussion"
    r.font.size = Pt(20); r.font.color.rgb = GREY; r.font.name = FONT

    prs.save(out_path)
    return out_path


if __name__ == "__main__":
    meta = {
        "institution": "Sri Sivasubramaniya Nadar College of Engineering",
        "department": "Department of Computer Science and Engineering",
        "title": "Hamiltonian-Encoded Quantum LSTM for Learning Higher-Order Dependencies "
                 "in Multivariate Time-Series Forecasting",
        "review": "First Review",
        "students_ppt": [
            "Student Name 1 (Register No.)",
            "Student Name 2 (Register No.)",
            "Student Name 3 (Register No.)",
        ],
        "supervisor": "Dr./Mr./Ms. Supervisor Name, Designation",
        "academic_year": "2026–2027",
    }
    out = os.path.join(HERE, "Review_Presentation_I.pptx")
    build(out, meta)
    print("Wrote", out)
