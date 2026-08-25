"""Classical LSTM baseline: same interface as QLSTMRegressor (src/qlstm_model.py),
gates replaced by a standard nn.LSTM. Used as the classical comparison point for
the QLSTM and (later) the H-QLSTM cell, trained under an identical harness."""
from torch import nn


class ClassicalLSTMRegressor(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_t, _) = self.lstm(x)
        return self.linear(h_t.squeeze(0)).squeeze(-1)
