import torch
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import ast
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
features = ['time', 'center_freq', 'dist', 'h_dist', 'v_dist', 'avgPower', 'avgSnr',
            'freq_offset', 'avg_pl', 'aod_theta', 'aoa_theta', 'aoa_phi',
            'pitch', 'yaw', 'roll', 'vel_x', 'vel_y', 'vel_z', 'speed', 
            'avg_pl_rolling', 'avg_pl_ewma']

# Load dataset
df = load_multiple_csvs("dataset/", features)
df.fillna(0, inplace=True)  # Handle NaNs

# Save min and max values for later denormalization
original_min = df.min()
original_max = df.max()
np.save("original_min.npy", original_min.values)
np.save("original_max.npy", original_max.values)
print("Saved min and max values for denormalization.")

# Normalize data
def normalize_data(df, min_vals, max_vals):
    return (df - min_vals) / (max_vals - min_vals + 1e-8)  # Avoid division by zero

df = normalize_data(df, original_min, original_max)

# Convert data into sequences (e.g., 50-timestep windows)
sequence_length = 50  # Adjust based on UAV flight duration
X_sequences = []

for i in range(len(df) - sequence_length):
    X_sequences.append(df.iloc[i:i + sequence_length].values)

X = torch.tensor(np.array(X_sequences), dtype=torch.float32)
y = torch.ones(X.shape[0], 1)  # Labels set to ones (for future use)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # No shuffling to maintain sequence order

# Define LSTM-based Generator
class Generator(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=128, num_layers=2):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.tanh = nn.Tanh()

    def forward(self, z):
        lstm_out, _ = self.lstm(z)
        output = self.fc(lstm_out)
        return self.tanh(output)

# Define LSTM-based Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim=128, num_layers=2):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Use the last LSTM output
        return self.sigmoid(output)

# GAN Training Function
def train_gan(generator, discriminator, dataloader, epochs=4000):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for real_data, _ in dataloader:
            batch_size = real_data.size(0)

            real_data = real_data.to(DEVICE)
            real_labels = torch.ones(batch_size, 1).to(DEVICE)
            fake_labels = torch.zeros(batch_size, 1).to(DEVICE)

            # Train Discriminator
            optimizer_D.zero_grad()
            outputs = discriminator(real_data)
            loss_real = criterion(outputs, real_labels)

            z = torch.randn(batch_size, sequence_length, len(features)).to(DEVICE)  # Generate time-dependent noise
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
        
        if epoch % 50 == 0:  # Print loss every 50 epochs
            print(f"Epoch {epoch+1}/{epochs}, D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}")

# Initialize and train GAN
generator = Generator(input_size=len(features), output_size=len(features)).to(DEVICE)
discriminator = Discriminator(input_size=len(features)).to(DEVICE)
train_gan(generator, discriminator, dataloader)

torch.save(generator.state_dict(), "generatorcheckpoint.pth")
torch.save(discriminator.state_dict(), "discriminatorcheckpoint.pth")
print("Saved model checkpoints.")

