import torch
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load CSV Data
def load_multiple_csvs(directory, features):
    dataframes = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, file), usecols=features)
            dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

# Define feature columns
features = ['center_freq', 'dist', 'h_dist', 'v_dist', 'avgPower', 'avgSnr',
            'freq_offset', 'avg_pl', 'aod_theta', 'aoa_theta', 'aoa_phi',
            'pitch', 'yaw', 'roll', 'vel_x', 'vel_y', 'vel_z', 'speed', 
            'avg_pl_rolling', 'avg_pl_ewma']

# Load dataset
df = load_multiple_csvs("dataset/", features)
df.fillna(0, inplace=True)  # Handle NaNs

original_min = df.min()
original_max = df.max()
np.save("original_min.npy", original_min.values)
np.save("original_max.npy", original_max.values)
print("Saved min and max values for denormalization.")




# Normalize data
def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min() + 1e-8)

df = normalize_data(df)

# Convert data into sequences (e.g., 50-timestep windows)
sequence_length = 1000  # Adjust based on UAV flight duration
X_sequences = []
for i in range(len(df) - sequence_length):
    X_sequences.append(df.iloc[i:i + sequence_length].values)

X = torch.tensor(np.array(X_sequences), dtype=torch.float32)
y = torch.ones(X.shape[0], 1)  # Dummy labels

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)



####################################
# Define Hybrid Generator (LSTM + CNN)
####################################
class Generator(nn.Module):
    def __init__(self, noise_dim, seq_len, output_dim, lstm_hidden=128, lstm_layers=1):
        """
        noise_dim: dimension of noise input (and LSTM input features)
        seq_len: sequence length (number of timesteps)
        output_dim: desired feature dimension of the output (e.g., len(features))
        """
        super(Generator, self).__init__()
        self.seq_len = seq_len
        
        # LSTM Block to capture sequence dependencies
        self.lstm = nn.LSTM(input_size=noise_dim, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True)
        # A linear projection to adapt LSTM output to output_dim
        self.fc = nn.Linear(lstm_hidden, output_dim)
        
        # CNN Block to refine generated sequence (assumes input shape: (batch, output_dim, seq_len))
        self.conv1 = nn.Conv1d(in_channels=output_dim, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.prelu1 = nn.PReLU()
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.prelu2 = nn.PReLU()
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=output_dim, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()  # Ensures output is within [-1, 1] range

    def forward(self, z):
        # z shape: (batch, seq_len, noise_dim)
        lstm_out, _ = self.lstm(z)  # (batch, seq_len, lstm_hidden)
        proj = self.fc(lstm_out)  # (batch, seq_len, output_dim)
        proj = proj.permute(0,2,1)        
        x = self.prelu1(self.bn1(self.conv1(proj)))
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.conv3(x)  # (batch, output_dim, seq_len)
        x = self.tanh(x)  # Apply tanh activation
        return x.permute(0,2,1)  # Transpose back to (batch, seq_len, output_dim)

####################################
# Define CNN-based Discriminator
####################################
class Discriminator(nn.Module):
    def __init__(self, input_size, seq_len, dropout_prob=0.1):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
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
        
        # Global Average Pooling instead of flattening all time steps
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0,2,1)
        # x shape: (batch, input_size, seq_len)
        x = self.dropout1(self.prelu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.prelu2(self.bn2(self.conv2(x))))
        x = self.dropout3(self.prelu3(self.bn3(self.conv3(x))))
        x = self.dropout4(self.prelu4(self.bn4(self.conv4(x))))
        
        x = self.global_avg_pool(x)  # (batch, 512, 1)
        x = x.squeeze(2)  # Remove last dimension -> (batch, 512)
        x = self.fc(x)  # Fully connected layer
        return self.sigmoid(x)  # (batch, 1) Probability score


####################################
# GAN Training Function
####################################
def train_gan(generator, discriminator, dataloader, epochs=500):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for real_data, _ in dataloader:
            batch_size = real_data.size(0)
            real_data = real_data.to(DEVICE)
            #real_labels = torch.ones(batch_size, 1).to(DEVICE)
            #fake_labels = torch.zeros(batch_size, 1).to(DEVICE)
            real_labels = torch.full((batch_size, 1), 0.9).to(DEVICE)
            fake_labels = torch.full((batch_size, 1), 0.1).to(DEVICE)

            
            # Train Discriminator
            optimizer_D.zero_grad()
            outputs = discriminator(real_data)
            loss_real = criterion(outputs, real_labels)
            
            # Generate noise: shape (batch, seq_len, noise_dim)
            z = torch.randn(batch_size, sequence_length, len(features)).to(DEVICE)
            fake_data = generator(z)
            outputs = discriminator(fake_data.detach())
            loss_fake = criterion(outputs, fake_labels)
            
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            outputs = discriminator(fake_data)
            loss_G = criterion(outputs, real_labels)
            loss_G.backward()
            optimizer_G.step()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}")

####################################
# Initialize and train GAN
####################################
# For text generation, output_dim could be the vocabulary size.
# Here we set noise_dim and output_dim to len(features) for demonstration.
noise_dim = len(features)
output_dim = len(features)

generator = Generator(noise_dim=noise_dim, seq_len=sequence_length, output_dim=output_dim).to(DEVICE)
discriminator = Discriminator(input_size=len(features), seq_len=sequence_length).to(DEVICE)
train_gan(generator, discriminator, dataloader)

torch.save(generator.state_dict(), "generatorcheckpoint.pth")
torch.save(discriminator.state_dict(), "discriminatorcheckpoint.pth")
print("Saved model checkpoints.")

