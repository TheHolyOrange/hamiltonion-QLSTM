"""
Quantum-specific analyses of the QLSTM trained on UniEload (run_unieload.py):

  barren     Barren-plateau scan: Var[dC/dtheta] of the QLSTM gate circuit
             (AngleEmbedding + CNOT/RX-RY-RZ ansatz) at random initialization
             vs. qubit count and depth. Cost C is the local observable
             <Z_0>, inputs are random. Exponential
             decay of the variance with n_qubits is the barren-plateau
             signature (McClean et al., Nat. Commun. 9, 4812 (2018)).
  landscape  Loss ("energy") landscape: validation MSE of the trained QLSTM on
             a 2D slice through its VQC parameter space,
                 L(a, b) = MSE(theta* + a*d1 + b*d2),
             with d1, d2 random directions normalized per gate to the norm of
             that gate's trained weights (filter normalization, Li et al.,
             NeurIPS 2018). Classical layers are held at their trained values.
  noise      Noise robustness: test RMSE of the trained QLSTM (no retraining)
             when every <Z> is estimated from a finite number of shots, and
             under per-gate depolarizing noise of strength p
             (QLSTM noise_p / shots, src/qlstm_model.py).

Usage: python -m src.quantum_analysis {barren,landscape,noise}
Artifacts: results/unieload/analysis/*.json and results/unieload/plots/*.png
"""
import argparse
import json
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch

OUT = "results/unieload"
ADIR = f"{OUT}/analysis"
PDIR = f"{OUT}/plots"


def _save(name, obj):
    os.makedirs(ADIR, exist_ok=True)
    with open(f"{ADIR}/{name}.json", "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------- barren ---

def _qlstm_circuit(nq, L):
    dev = qml.device("default.qubit", wires=nq)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def c(x, w):
        qml.AngleEmbedding(x, wires=range(nq))
        for l in range(L):
            for k in (1, 2):
                for j in range(nq):
                    qml.CNOT([j, (j + k) % nq])
            for i in range(nq):
                qml.RX(w[l, 0, i], i); qml.RY(w[l, 1, i], i); qml.RZ(w[l, 2, i], i)
        return qml.expval(qml.PauliZ(0))

    def sample(gen):
        x = torch.randn(nq, generator=gen, dtype=torch.float64)
        w = (torch.rand(L, 3, nq, generator=gen, dtype=torch.float64) * 2 * np.pi).requires_grad_()
        return c, (x, w), [w], nq  # tracked param: layer-0 RY on wire 0 = flat index nq
    return sample


# starts at 3: the QLSTM ansatz CNOTs each wire to wires +1 and +2 away, which
# is a self-CNOT (undefined) at n_qubits=2
def barren(qubits=(3, 4, 6, 8, 10), depths=(1, 2, 4, 8), n_samples=200):
    # "var": Var over random inits of dC/dtheta for ONE fixed parameter (the
    # layer-0 RY on wire 0), the standard barren-plateau quantity.
    # "var_mean_all": the same variance averaged over all rotation parameters
    # -- reported for completeness, but it is diluted by parameters that cannot
    # affect <Z_0> at all (e.g. the final-layer RZs commute with the readout),
    # whose share grows with n_qubits, so it decays with width even without a
    # plateau.
    res = {"circuit": "QLSTM gate VQC", "cost": "<Z_0>", "n_samples": n_samples,
           "qubits": list(qubits), "depths": list(depths), "var": {}, "var_mean_all": {}}
    for L in depths:
        row, row_all = [], []
        for nq in qubits:
            sample = _qlstm_circuit(nq, L)
            gen = torch.Generator().manual_seed(1234)
            grads = []
            t0 = time.time()
            for _ in range(n_samples):
                c, args, params, k = sample(gen)
                out = c(*args)
                g = torch.autograd.grad(out, params)
                grads.append(torch.cat([gi.flatten() for gi in g]).detach().numpy())
            grads = np.stack(grads)                  # (samples, params)
            v = float(grads[:, k].var())
            v_all = float(grads.var(axis=0).mean())
            row.append(v); row_all.append(v_all)
            print(f"[barren] QLSTM ansatz       L={L} n={nq:2d}  Var[dC/dtheta_0]={v:.3e}  "
                  f"mean-over-params={v_all:.3e}  ({time.time()-t0:.1f}s)", flush=True)
        res["var"][str(L)] = row
        res["var_mean_all"][str(L)] = row_all
    finish_barren(res)


def finish_barren(res):
    """Fit, save and plot a completed barren-plateau scan."""
    qubits, depths = res["qubits"], res["depths"]
    # exponential decay rate per depth, Var ~ b^-n
    res["decay_fit"] = {}
    for L in depths:
        slope = float(np.polyfit(qubits, np.log(res["var"][str(L)]), 1)[0])
        res["decay_fit"][str(L)] = {"log_var_slope_per_qubit": slope, "base_b": float(np.exp(-slope))}
    _save("barren_plateau", res)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for L in depths:
        axes[0].semilogy(qubits, res["var"][str(L)], "o-", label=f"{L} layer{'s' if L > 1 else ''}")
    axes[0].set_xlabel("number of qubits"); axes[0].set_ylabel("Var[∂C/∂θ₀]  (C = <Z₀>, random init)")
    axes[0].set_title("Gradient variance vs. width"); axes[0].grid(alpha=0.3); axes[0].legend()
    for i, nq in enumerate(qubits):
        axes[1].semilogy(depths, [res["var"][str(L)][i] for L in depths], "o-", label=f"{nq} qubits")
    axes[1].set_xlabel("number of ansatz layers"); axes[1].set_xscale("log", base=2)
    axes[1].set_title("Gradient variance vs. depth"); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle(f"QLSTM VQC barren-plateau scan ({res['n_samples']} random inits per point)")
    fig.tight_layout(); os.makedirs(PDIR, exist_ok=True)
    fig.savefig(f"{PDIR}/qlstm_barren_plateau.png", dpi=150); plt.close(fig)


# -------------------------------------------------------- trained model ---

def _trained_qlstm(**kw):
    from run_unieload import N_QUBITS, N_QLAYERS, HIDDEN_SIZE
    from src.qlstm_model import QLSTMRegressor
    from src.unieload_preprocessing import FEATURE_COLS
    m = QLSTMRegressor(len(FEATURE_COLS), HIDDEN_SIZE, N_QUBITS, N_QLAYERS, **kw)
    m.load_state_dict(torch.load(f"{OUT}/checkpoints/qlstm_best.pt", map_location="cpu"))
    return m.eval()


def _mse(model, X, y):
    with torch.no_grad():
        return float(torch.mean((model(X) - y) ** 2))


def _stack(ds, n=None):
    idx = range(len(ds) if n is None else min(n, len(ds)))
    X = torch.stack([ds[i][0] for i in idx]); y = torch.stack([ds[i][1] for i in idx])
    return X, y


def landscape(grid=13, span=1.0, n_windows=160):
    from run_unieload import load_data
    _, val_ds, _, _ = load_data()
    X, y = _stack(val_ds, n_windows)
    model = _trained_qlstm()
    vqc = {k: p for k, p in model.named_parameters() if ".VQC." in k}
    theta = {k: p.detach().clone() for k, p in vqc.items()}
    gen = torch.Generator().manual_seed(7)
    dirs = []
    for _ in range(2):
        d = {}
        for k, t in theta.items():
            r = torch.randn(t.shape, generator=gen)
            d[k] = r / r.norm() * t.norm()  # per-gate filter normalization
        dirs.append(d)

    alphas = np.linspace(-span, span, grid)
    Z = np.zeros((grid, grid))
    t0 = time.time()
    for i, a in enumerate(alphas):
        for j, b in enumerate(alphas):
            with torch.no_grad():
                for k, p in vqc.items():
                    p.copy_(theta[k] + a * dirs[0][k] + b * dirs[1][k])
            Z[i, j] = _mse(model, X, y)
        print(f"[landscape] row {i+1}/{grid} done  min so far {Z[:i+1].min():.4f}  ({time.time()-t0:.0f}s)", flush=True)
    c = grid // 2
    _save("loss_landscape", {"alphas": alphas.tolist(), "val_mse_scaled": Z.tolist(),
                             "center_mse": Z[c, c], "n_windows": n_windows, "span": span,
                             "n_vqc_params": int(sum(t.numel() for t in theta.values()))})

    A, B = np.meshgrid(alphas, alphas, indexing="ij")
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    cs = ax1.contourf(A, B, Z, levels=30, cmap="viridis")
    ax1.contour(A, B, Z, levels=15, colors="k", linewidths=0.3)
    ax1.plot(0, 0, "r*", ms=12, label="trained θ*")
    fig.colorbar(cs, ax=ax1, label="val MSE (scaled)")
    ax1.set_xlabel("direction 1 (α)"); ax1.set_ylabel("direction 2 (β)"); ax1.legend()
    ax1.set_title("Loss landscape around trained QLSTM (VQC params)")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(A, B, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax2.set_xlabel("α"); ax2.set_ylabel("β"); ax2.set_zlabel("val MSE")
    ax2.set_title("Energy (loss) surface")
    fig.tight_layout(); fig.savefig(f"{PDIR}/qlstm_loss_landscape.png", dpi=150); plt.close(fig)


def noise(n_windows=120, shots=(100, 1000, 10000, 100000), ps=(0.001, 0.005, 0.01, 0.02)):
    from run_unieload import load_data
    from src.utils import rmse
    from src.unieload_preprocessing import inverse_transform_target
    _, _, test_ds, meta = load_data()
    X, y = _stack(test_ds, n_windows)
    nf = len(meta["feature_cols"])
    to_kw = lambda v: inverse_transform_target(np.asarray(v), meta["scaler"], meta["target_idx"], nf)
    y_kw = to_kw(y.numpy())

    def run(**kw):
        t0 = time.time()
        torch.manual_seed(0)
        with torch.no_grad():
            p = _trained_qlstm(**kw)(X).numpy()
        r = rmse(y_kw, to_kw(p))
        print(f"[noise] {kw or 'noiseless'}  RMSE={r:.3f} kW  ({time.time()-t0:.0f}s)", flush=True)
        return r

    res = {"n_test_windows": n_windows, "noiseless_rmse_kw": run(), "shots": {}, "depolarizing": {}}
    for s in shots:
        res["shots"][str(s)] = run(shots=s)
    for p in ps:
        res["depolarizing"][str(p)] = run(noise_p=p)
    _save("noise_robustness", res)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    base = res["noiseless_rmse_kw"]
    axes[0].semilogx(shots, [res["shots"][str(s)] for s in shots], "o-")
    axes[0].axhline(base, ls="--", c="gray", label="noiseless (exact <Z>)")
    axes[0].set_xlabel("shots per expectation value"); axes[0].set_ylabel("test RMSE (kW)")
    axes[0].set_title("Shot noise"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot([0] + list(ps), [base] + [res["depolarizing"][str(p)] for p in ps], "o-", c="C3")
    axes[1].axhline(base, ls="--", c="gray")
    axes[1].set_xlabel("depolarizing probability p (per gate, per wire)")
    axes[1].set_title("Depolarizing gate noise"); axes[1].grid(alpha=0.3)
    fig.suptitle(f"Trained QLSTM under quantum noise (UniEload test, first {n_windows} h, no retraining)")
    fig.tight_layout(); fig.savefig(f"{PDIR}/qlstm_noise_robustness.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["barren", "landscape", "noise"])
    {"barren": barren, "landscape": landscape, "noise": noise}[ap.parse_args().what]()
