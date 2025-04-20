import torch
import torch.nn as nn

# --- Generator-specific variables ---
LSTM_HIDDEN_G = 256
LSTM_LAYERS_G = 1

class Generator(nn.Module):
    # Takes noise_dim + condition_dim as input size
    def __init__(self, generator_input_dim, seq_len, output_dim, # Note: output_dim is target_dim
                lstm_hidden=LSTM_HIDDEN_G, lstm_layers=LSTM_LAYERS_G):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size=generator_input_dim, # Changed input size
                            hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True)
        # FC layer maps LSTM output to the target dimension (avg_pl)
        self.fc = nn.Linear(lstm_hidden, output_dim)
        # CNN layers operate on the target dimension
        self.conv1 = nn.Conv1d(in_channels=output_dim, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=output_dim, kernel_size=3, padding=1)
        self.tanh = nn.Tanh() # Output is [-1, 1]

    def forward(self, z_cond): # Input is concatenated noise and condition
        # z_cond shape: (batch, seq_len, generator_input_dim)
        lstm_out, _ = self.lstm(z_cond) # (batch, seq_len, lstm_hidden)
        proj = self.fc(lstm_out)   # (batch, seq_len, output_dim - e.g., just avg_pl)
        proj = proj.permute(0, 2, 1) # (batch, output_dim, seq_len) for Conv1D
        x = self.prelu1(self.bn1(self.conv1(proj)))
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = self.tanh(x) # Output avg_pl in [-1, 1]
        return x.permute(0, 2, 1) # (batch, seq_len, output_dim)
    