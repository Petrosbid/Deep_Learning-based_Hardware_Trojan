#
# model.py
# (Phase 3: Define LSTM model architecture)
#
import torch
import torch.nn as nn


class TrojanLSTM(nn.Module):
    """
    Implementation of LSTM architecture according to the paper.
    """

    def __init__(self, input_size=100, hidden_size=128, num_layers=2, output_size=2):
        super(TrojanLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,  # 100 (from embedding)
            hidden_size=hidden_size,  # 128
            num_layers=num_layers,  # 2
            batch_first=True  # Input is (Batch, SeqLen, Features)
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x.shape = (batch_size, MAX_TRACE_LENGTH, 100)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # h_n.shape = (num_layers, batch_size, hidden_size)
        # h_n[-1] = last layer
        last_hidden_state = h_n[-1]  # shape = (batch_size, 128)

        # Pass final output through FC layer
        out = self.fc(last_hidden_state)  # shape = (batch_size, 2)

        return out