import torch
import torch.nn as nn

class Critic(nn.Module):
    # Input size is the dimension of the target features ONLY
    def __init__(self, target_dim, condition_dim, seq_len, dropout_prob=0.1):
        super(Critic, self).__init__()
        # Conv layers operate on the target dimension (e.g., 1 for avgSnr)
        total_input_dim = target_dim + condition_dim
        self.conv1 = nn.Conv1d(in_channels=total_input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout_prob)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.prelu3 = nn.PReLU()
        self.dropout3 = nn.Dropout(dropout_prob)

        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.prelu4 = nn.PReLU()
        self.dropout4 = nn.Dropout(dropout_prob)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 1)
        # NO Sigmoid/Tanh

    def forward(self, x_target, x_condition): # Input is only the target sequence (real or fake avg_pl)
        # x_target shape: (batch, seq_len, target_dim)
        x = torch.cat([x_target, x_condition], dim=2)
        x = x.permute(0, 2, 1) # (batch, target_dim, seq_len) for Conv1D
        x = self.dropout1(self.prelu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.prelu2(self.bn2(self.conv2(x))))
        x = self.dropout3(self.prelu3(self.bn3(self.conv3(x))))
        x = self.dropout4(self.prelu4(self.bn4(self.conv4(x))))
        x = self.global_avg_pool(x).squeeze(2) # (batch, 512)
        score = self.fc(x) # Raw score
        return score # (batch, 1)
    