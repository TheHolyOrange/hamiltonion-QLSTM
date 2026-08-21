"""
QLSTM: a quantum-enhanced LSTM cell where each classical gate (forget, input,
update/candidate, output) is replaced by a variational quantum circuit (VQC).

Architecture (adapted from the SPP-QLSTM reference implementation
https://github.com/QCL-PKNU/SPP-QLSTM, generalized to arbitrary feature counts
and batched execution):

  concat(h_{t-1}, x_t) --Linear--> y_t (dim = n_qubits)
  y_t --AngleEmbedding + entangling/rotation ansatz (VQC)--> quantum expectations
  quantum expectations --Linear--> gate pre-activation (dim = hidden_size)
  gate pre-activation --sigmoid/tanh--> f_t, i_t, g_t, o_t
  c_t = f_t * c_{t-1} + i_t * g_t
  h_t = o_t * tanh(c_t)

This is the classical-VQC-hybrid baseline: the current (30%) checkpoint. The
explicit dynamical-Hamiltonian data encoding described in the project's
problem statement (embedding the input into the generator of time evolution
itself, e.g. exp(-i H(x) t), rather than a fixed data-reuploading ansatz) is
planned as the next phase and is not yet implemented here.
"""
import torch
from torch import nn
import pennylane as qml


class QLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits=4,
        n_qlayers=1,
        n_vrotations=3,
        batch_first=True,
        backend="default.qubit",
    ):
        super().__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = input_size + hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.n_vrotations = n_vrotations
        self.batch_first = batch_first

        self.wires_forget = [f"wf{i}" for i in range(n_qubits)]
        self.wires_input = [f"wi{i}" for i in range(n_qubits)]
        self.wires_update = [f"wu{i}" for i in range(n_qubits)]
        self.wires_output = [f"wo{i}" for i in range(n_qubits)]

        dev_forget = qml.device(backend, wires=self.wires_forget)
        dev_input = qml.device(backend, wires=self.wires_input)
        dev_update = qml.device(backend, wires=self.wires_update)
        dev_output = qml.device(backend, wires=self.wires_output)

        def ansatz(params, wires_type):
            for i in range(1, 3):
                for j in range(self.n_qubits):
                    tgt = (j + i) % self.n_qubits
                    qml.CNOT(wires=[wires_type[j], wires_type[tgt]])
            for i in range(self.n_qubits):
                qml.RX(params[0][i], wires=wires_type[i])
                qml.RY(params[1][i], wires=wires_type[i])
                qml.RZ(params[2][i], wires=wires_type[i])

        def VQC(features, weights, wires_type):
            qml.templates.AngleEmbedding(features, wires=wires_type)
            qml.layer(ansatz, self.n_qlayers, weights, wires_type=wires_type)

        def make_circuit(wires_type, dev):
            def _circuit(inputs, weights):
                VQC(inputs, weights, wires_type)
                return [qml.expval(qml.PauliZ(w)) for w in wires_type]
            return qml.QNode(_circuit, dev, interface="torch")

        weight_shapes = {"weights": (n_qlayers, n_vrotations, n_qubits)}

        self.clayer_in = nn.Linear(self.concat_size, n_qubits)
        self.VQC = nn.ModuleDict({
            "forget": qml.qnn.TorchLayer(make_circuit(self.wires_forget, dev_forget), weight_shapes),
            "input": qml.qnn.TorchLayer(make_circuit(self.wires_input, dev_input), weight_shapes),
            "cand": qml.qnn.TorchLayer(make_circuit(self.wires_update, dev_update), weight_shapes),
            "output": qml.qnn.TorchLayer(make_circuit(self.wires_output, dev_output), weight_shapes),
        })
        self.clayer_out = nn.Linear(n_qubits, hidden_size)

    def forward(self, x, init_states=None):
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)

        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t, c_t = init_states

        hidden_seq = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            v_t = torch.cat((h_t, x_t), dim=1)
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(self.clayer_out(self.VQC["forget"](y_t)))
            i_t = torch.sigmoid(self.clayer_out(self.VQC["input"](y_t)))
            g_t = torch.tanh(self.clayer_out(self.VQC["cand"](y_t)))
            o_t = torch.sigmoid(self.clayer_out(self.VQC["output"](y_t)))

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(1))

        hidden_seq = torch.cat(hidden_seq, dim=1)
        return hidden_seq, (h_t, c_t)


class QLSTMRegressor(nn.Module):
    """QLSTM followed by a linear head -> single-step regression (next OT value)."""

    def __init__(self, num_features, hidden_size, n_qubits=4, n_qlayers=1, n_vrotations=3):
        super().__init__()
        self.qlstm = QLSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
            n_vrotations=n_vrotations,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_t, _) = self.qlstm(x)
        return self.linear(h_t).squeeze(-1)


if __name__ == "__main__":
    import time
    model = QLSTMRegressor(num_features=11, hidden_size=8, n_qubits=4, n_qlayers=1)
    x = torch.randn(16, 24, 11)
    t0 = time.time()
    out = model(x)
    print("output shape:", out.shape, "forward time:", time.time() - t0)
    loss = out.sum()
    t0 = time.time()
    loss.backward()
    print("backward time:", time.time() - t0)
    n_params = sum(p.numel() for p in model.parameters())
    print("total trainable params:", n_params)
