#
# model.py
# (Phase 3: LSTM Model Architecture Definition)
#
import torch
import torch.nn as nn


class TrojanLSTM(nn.Module):
    """
    LSTM architecture implementation as per the paper.
    """

    def __init__(self, input_size=100, hidden_size=128, num_layers=2, output_size=2):
        super(TrojanLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. LSTM Layer
        # As per paper: 2 LSTM layers
        # As per paper: 128 hidden units (hidden_size)
        self.lstm = nn.LSTM(
            input_size=input_size,  # 100 (from embedding)
            hidden_size=hidden_size,  # 128
            num_layers=num_layers,  # 2
            batch_first=True  # Input is (Batch, SeqLen, Features)
        )

        # 2. Fully Connected (FC) Layer
        # Maps LSTM output (128) to 2 classes (Normal/HT)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x.shape = (batch_size, MAX_TRACE_LENGTH, 100)

        # Initialize hidden and cell states (h0, c0) with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass input through LSTM
        # lstm_out.shape = (batch_size, MAX_TRACE_LENGTH, 128)
        # (h_n, c_n) = final hidden and cell states
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # We only need the output of the last LSTM layer at the final time step
        # h_n.shape = (num_layers, batch_size, hidden_size)
        # h_n[-1] = last layer
        last_hidden_state = h_n[-1]  # shape = (batch_size, 128)

        # Pass final output through FC layer
        out = self.fc(last_hidden_state)  # shape = (batch_size, 2)

        return out