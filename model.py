#
# model.py
# (فاز 3: تعریف معماری مدل LSTM)
#
import torch
import torch.nn as nn


class TrojanLSTM(nn.Module):
    """
    پیاده‌سازی معماری LSTM مطابق با مقاله.
    """

    def __init__(self, input_size=100, hidden_size=128, num_layers=2, output_size=2):
        super(TrojanLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. لایه LSTM
        # مطابق مقاله: 2 لایه LSTM
        # مطابق مقاله: 128 واحد مخفی (hidden_size)
        self.lstm = nn.LSTM(
            input_size=input_size,  # 100 (از embedding)
            hidden_size=hidden_size,  # 128
            num_layers=num_layers,  # 2
            batch_first=True  # ورودی (Batch, SeqLen, Features) است
        )

        # 2. لایه Fully Connected (FC)
        # خروجی LSTM (128) را به 2 کلاس (Normal/HT) نگاشت می‌کند
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x.shape = (batch_size, MAX_TRACE_LENGTH, 100)

        # مقداردهی اولیه حالت‌های مخفی و سلولی (h0, c0) با صفر
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # عبور دادن ورودی از LSTM
        # lstm_out.shape = (batch_size, MAX_TRACE_LENGTH, 128)
        # (h_n, c_n) = آخرین حالت‌های مخفی و سلولی
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # ما فقط به خروجی آخرین لایه LSTM در آخرین مرحله زمانی نیاز داریم
        # h_n.shape = (num_layers, batch_size, hidden_size)
        # h_n[-1] = آخرین لایه
        last_hidden_state = h_n[-1]  # shape = (batch_size, 128)

        # عبور دادن خروجی نهایی از لایه FC
        out = self.fc(last_hidden_state)  # shape = (batch_size, 2)

        return out