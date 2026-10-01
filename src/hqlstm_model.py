"""
H-QLSTM: the dynamical-Hamiltonian extension of QLSTM (src/qlstm_model.py).

Where QLSTM embeds classical features into a *fixed* data-reuploading ansatz
(AngleEmbedding followed by a trainable rotation/CNOT block), this module
implements the encoding the project's problem statement and report
(Section 6) describe as the intended next phase: embedding the input into
the *generator* of time evolution itself,

    U(x, t) = T exp(-i * integral H(x(t)) dt)

rather than into fixed gate angles. Concretely, at each of the four LSTM
gates and for each of n_qlayers layers l:

    H_l(x) = sum_i x_i        * Z_i            (data-dependent local field --
                                                  this is what makes the
                                                  *generator itself* a function
                                                  of the input, not just a
                                                  rotation angle)
           + sum_i alpha_{l,i} * X_i            (trainable transverse field --
                                                  does not commute with the Z
                                                  terms, so evolution actually
                                                  mixes the computational basis
                                                  rather than only accumulating
                                                  phases)
           + sum_i beta_{l,i}  * Z_i Z_{i+1}    (trainable entangling coupling)

and qml.ApproxTimeEvolution(H_l, time_l, n=1) Trotter-steps exp(-i H_l time_l)
onto the qubits. Layering n_qlayers such (non-commuting, since alpha/beta
differ per layer) evolutions is a first-order-Trotterized approximation to
the time-ordered exponential of a piecewise-constant H(t); composed further
across the LSTM's own time recurrence (a fresh H_l per time step t, since the
Z-field term depends on x_t), this is what gives the architecture algebraic
access to the Dyson/Magnus cross terms (coupling multiple distinct time
points through the non-commutativity of H at different times) that a
fixed-ansatz VQC lacks by construction -- see README.md and
report/QLSTM_Status_Report.pdf Sections 2 and 6 for the full argument.

Everything outside VQC construction (linear in/out projections, gate
nonlinearities, cell/hidden state update, CPU-pinning rationale for the
PennyLane simulator) is unchanged from QLSTM, so the two are a controlled
comparison of encoding scheme alone, at matched hidden size / qubit count /
layer count.
"""
import torch
from torch import nn
import pennylane as qml


class HQLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits=4,
        n_qlayers=1,
        n_trotter_steps=2,
        batch_first=True,
        backend="default.qubit",
    ):
        super().__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = input_size + hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.n_trotter_steps = n_trotter_steps
        self.batch_first = batch_first

        self.wires_forget = list(range(n_qubits))
        self.wires_input = list(range(n_qubits))
        self.wires_update = list(range(n_qubits))
        self.wires_output = list(range(n_qubits))

        dev_forget = qml.device(backend, wires=n_qubits)
        dev_input = qml.device(backend, wires=n_qubits)
        dev_update = qml.device(backend, wires=n_qubits)
        dev_output = qml.device(backend, wires=n_qubits)

        n_q = self.n_qubits

        def build_hamiltonian(x, alpha_l, beta_l, gamma_l):
            """x: (batch, n_qubits) classical features (already tanh-bounded
            in forward(), see note there) -> data-dependent generator
            coefficients. alpha_l, beta_l, gamma_l: (n_qubits,) trainable
            weights for this layer. gamma_l is a trainable local Z-field
            bias added alongside the data term -- without it, the only
            trainable weights touching the Z-basis come indirectly through
            beta's ZZ coupling, which under-parameterizes this circuit
            relative to QLSTM's per-qubit RZ rotation (src/qlstm_model.py):
            QLSTM has 3 trainable angles/qubit/layer (RX, RY, RZ) vs. this
            circuit's 2 (alpha, beta) without gamma. Adding gamma brings the
            trainable-weight count to parity (3/qubit/layer: alpha, beta,
            gamma) so the two circuits are capacity-matched, not just
            qubit/layer-count-matched. Returns a qml.Hamiltonian with batched
            (per-sample) coefficients."""
            batch = x.shape[0]
            coeffs, ops = [], []
            for i in range(n_q):
                coeffs.append(x[:, i] + gamma_l[i].expand(batch))
                ops.append(qml.PauliZ(i))
            for i in range(n_q):
                coeffs.append(alpha_l[i].expand(batch))
                ops.append(qml.PauliX(i))
            for i in range(n_q):
                j = (i + 1) % n_q
                coeffs.append(beta_l[i].expand(batch))
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
            return qml.Hamiltonian(torch.stack(coeffs), ops)

        def make_circuit(wires_type, dev):
            def _circuit(inputs, alpha, beta, gamma, time):
                for l in range(self.n_qlayers):
                    H_l = build_hamiltonian(inputs, alpha[l], beta[l], gamma[l])
                    # n_trotter_steps > 1 Trotter-steps the same total
                    # evolution time into finer sub-steps, which lowers the
                    # first-order Trotter error (O(time^2 / n) per layer)
                    # without changing trainable-parameter count -- a purely
                    # numerical-fidelity knob, tuned separately from model
                    # capacity.
                    qml.ApproxTimeEvolution(H_l, time[l], self.n_trotter_steps)
                return [qml.expval(qml.PauliZ(w)) for w in wires_type]
            return qml.QNode(_circuit, dev, interface="torch")

        weight_shapes = {
            "alpha": (n_qlayers, n_qubits),
            "beta": (n_qlayers, n_qubits),
            "gamma": (n_qlayers, n_qubits),
            "time": (n_qlayers,),
        }

        # qml.qnn.TorchLayer's default init draws every weight from
        # Uniform(0, 2*pi) -- correct for a *rotation angle* (periodic, meant
        # to stand alone in a gate), but wrong here: `time` is a Trotter-step
        # *duration* that gets multiplied by `alpha`/`beta`/`x` before
        # Trotterization, and qml.ApproxTimeEvolution(H, time, n=1) is only a
        # valid first-order approximation of exp(-i*H*time) when
        # time * ||H|| is bounded. With n_qubits=6 and alpha/beta/time all
        # drawn independently from [0, 2*pi), the effective rotation angle at
        # init is up to (2*pi)**2 * n_qubits ~= 250 rad -- deep in an aliased,
        # per-step-discontinuous regime that a single Trotter step badly
        # mis-approximates: measured gradient norm w.r.t. alpha/beta/time at
        # init is large (~26) *and* unstable (+/-9 std across random draws),
        # which is consistent with the erratic epoch-to-epoch val-loss swings
        # seen in training (each Adam step can jump the effective angle by a
        # full alias, landing in an unrelated part of the periodic landscape).
        #
        # The fix keeps alpha/beta at a modest O(1) coefficient scale and
        # `time` at a fixed sub-1 duration, rather than letting *both*
        # multiplicands span a full 2*pi turn. Going too far the other way
        # (time near 0) is just as broken in a different way: the circuit
        # starts at exp(-i*H*0) = identity, and since the readout is
        # expval(Z) on a |0>-initialized register, d/d(angle)[cos(angle)] = 0
        # at angle = 0 -- a flat point that kills gradients for every gate
        # simultaneously (verified empirically: grad norm ~0.18 vs. ~26 for
        # the original init, and training stalled at train_loss~1.0 for two
        # full epochs when tried). Uniform(0.3, 0.6) for `time` was chosen by
        # sweeping the actual circuit's gradient norm (not by looking at
        # downstream val/test loss): it gives a ~5x smaller and far more
        # stable gradient than the original init (mean 5.8, std 1.5 vs.
        # mean 26, std 9 over 12 random draws) while keeping expval(Z) well
        # away from its saturated +/-1 plateau (spread ~0.2, vs. ~0 at
        # time ~ 0).
        init_method = {
            "alpha": lambda t: nn.init.uniform_(t, -1.0, 1.0),
            "beta": lambda t: nn.init.uniform_(t, -1.0, 1.0),
            "gamma": lambda t: nn.init.uniform_(t, -1.0, 1.0),
            "time": lambda t: nn.init.uniform_(t, 0.3, 0.6),
        }

        self.clayer_in = nn.Linear(self.concat_size, n_qubits)
        self.VQC = nn.ModuleDict({
            "forget": qml.qnn.TorchLayer(make_circuit(self.wires_forget, dev_forget), weight_shapes, init_method=init_method),
            "input": qml.qnn.TorchLayer(make_circuit(self.wires_input, dev_input), weight_shapes, init_method=init_method),
            "cand": qml.qnn.TorchLayer(make_circuit(self.wires_update, dev_update), weight_shapes, init_method=init_method),
            "output": qml.qnn.TorchLayer(make_circuit(self.wires_output, dev_output), weight_shapes, init_method=init_method),
        })
        self.clayer_out = nn.Linear(n_qubits, hidden_size)
        self._compute_device = torch.device("cpu")

    # See QLSTM.PREFERRED_DEVICE (src/qlstm_model.py) for the full rationale:
    # PennyLane's default.qubit simulator only runs on CPU, and at these qubit
    # counts there is no GPU benefit even where it's possible -- the cost is
    # per-circuit Python/Trotter-gate dispatch overhead, not statevector FLOPs.
    PREFERRED_DEVICE = "cpu"

    def _apply(self, fn, recurse=True):
        """See QLSTM._apply (src/qlstm_model.py): moves only the classical
        wrapper layers, leaving the VQC ModuleDict (PennyLane TorchLayers,
        which must stay on CPU) untouched. Data is shuttled cpu<->device
        around each VQC call in forward() below."""
        self.clayer_in._apply(fn, recurse)
        self.clayer_out._apply(fn, recurse)
        self._compute_device = fn(torch.zeros(1)).device
        return self

    def forward(self, x, init_states=None):
        device = self._compute_device
        x = x.to(device)
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)

        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_t, c_t = init_states

        hidden_seq = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            v_t = torch.cat((h_t, x_t), dim=1)
            # tanh-bound the Z-field data coefficient to the same O(1) scale
            # as alpha/beta/gamma (see the init_method note above): clayer_in
            # is an unconstrained Linear, and an empirical trace of a fully
            # trained (pre-fix) checkpoint showed this coefficient drifting
            # from ~1.65 up to a ~2.0 plateau over the recurrence as training
            # progressed -- not yet the catastrophic aliasing the original
            # alpha/beta/time init hit, but it erodes the n=1 Trotter
            # approximation's validity the same way and tracked with a late
            # training val-loss spike (0.04 -> 0.25) right before early
            # stopping. Bounding it keeps time * ||H|| well-conditioned for
            # the whole run, not just at init.
            y_t = torch.tanh(self.clayer_in(v_t))
            y_t_cpu = y_t.cpu()  # VQC (default.qubit) simulates on CPU only

            f_t = torch.sigmoid(self.clayer_out(self.VQC["forget"](y_t_cpu).to(device)))
            i_t = torch.sigmoid(self.clayer_out(self.VQC["input"](y_t_cpu).to(device)))
            g_t = torch.tanh(self.clayer_out(self.VQC["cand"](y_t_cpu).to(device)))
            o_t = torch.sigmoid(self.clayer_out(self.VQC["output"](y_t_cpu).to(device)))

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(1))

        hidden_seq = torch.cat(hidden_seq, dim=1)
        return hidden_seq, (h_t, c_t)


class HQLSTMRegressor(nn.Module):
    """HQLSTM followed by a linear head -> single-step regression (next OT value)."""

    PREFERRED_DEVICE = "cpu"  # see HQLSTM.PREFERRED_DEVICE

    def __init__(self, num_features, hidden_size, n_qubits=4, n_qlayers=1, n_trotter_steps=2):
        super().__init__()
        self.hqlstm = HQLSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
            n_trotter_steps=n_trotter_steps,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_t, _) = self.hqlstm(x)
        return self.linear(h_t).squeeze(-1)


if __name__ == "__main__":
    import time
    model = HQLSTMRegressor(num_features=11, hidden_size=8, n_qubits=6, n_qlayers=2)
    x = torch.randn(32, 24, 11)
    t0 = time.time()
    out = model(x)
    print("output shape:", out.shape, "forward time:", time.time() - t0)
    loss = out.sum()
    t0 = time.time()
    loss.backward()
    print("backward time:", time.time() - t0)
    n_params = sum(p.numel() for p in model.parameters())
    print("total trainable params:", n_params)
