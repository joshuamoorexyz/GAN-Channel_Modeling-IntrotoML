import torch
import pandas as pd
import numpy as np

# Define Hybrid Generator (LSTM + CNN)
class Generator(torch.nn.Module):
    def __init__(self, noise_dim, seq_len, output_dim, lstm_hidden=128, lstm_layers=1):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.lstm = torch.nn.LSTM(input_size=noise_dim, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.fc = torch.nn.Linear(lstm_hidden, output_dim)

        self.conv1 = torch.nn.Conv1d(in_channels=output_dim, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.prelu1 = torch.nn.PReLU()

        self.conv2 = torch.nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.prelu2 = torch.nn.PReLU()

        self.conv3 = torch.nn.Conv1d(in_channels=64, out_channels=output_dim, kernel_size=3, padding=1)
        self.tanh = torch.nn.Tanh()

    def forward(self, z):
        lstm_out, _ = self.lstm(z)
        proj = self.fc(lstm_out)
        proj = proj.permute(0, 2, 1)
        x = self.prelu1(self.bn1(self.conv1(proj)))
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = self.tanh(x)
        return x.permute(0, 2, 1)

# Use this function in inference.py
def load_generator(model_path, noise_dim, seq_len, output_dim, device="cpu"):
    generator = Generator(noise_dim, seq_len, output_dim).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    return generator




# Denormalization function (assuming min-max scaling during training)
def denormalize_data(generated_data, original_min, original_max):
    return (generated_data + 1) / 2 * (original_max - original_min) + original_min

# Generate synthetic time-series data
def generate_data(generator, num_samples=100, sequence_length=50, input_size=20, device="cpu"):
    noise = torch.randn(num_samples, sequence_length, input_size, device=device)  # Now using LSTM input format
    with torch.no_grad():
        generated_data = generator(noise).cpu().numpy()
    return generated_data

# Save generated data to CSV with feature names
def save_to_csv(data, filename="generated_data.csv"):
    features = ['center_freq', 'dist', 'h_dist', 'v_dist', 'avgPower', 'avgSnr',
                'freq_offset', 'avg_pl', 'aod_theta', 'aoa_theta', 'aoa_phi',
                'pitch', 'yaw', 'roll', 'vel_x', 'vel_y', 'vel_z', 'speed', 
                'avg_pl_rolling', 'avg_pl_ewma']
    
    # Reshape data to 2D (if necessary)
    num_samples, sequence_length, num_features = data.shape
    reshaped_data = data.reshape(num_samples * sequence_length, num_features)

    df = pd.DataFrame(reshaped_data, columns=features)
    df.to_csv(filename, index=False)
    print(f"Saved generated data to {filename}")

if __name__ == "__main__":
    model_path = "generatorcheckpoint.pth"
    num_samples = 1000
    sequence_length = 50  # Must match training
    input_size = 20
    output_size = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load original min/max values used during training
    original_min = np.load("original_min.npy")  
    original_max = np.load("original_max.npy")  

    # Load trained generator
    generator = load_generator("generatorcheckpoint.pth", noise_dim=20, seq_len=50, output_dim=20, device="cuda")


    # Generate synthetic UAV flight data
    generated_data = generate_data(generator, num_samples, sequence_length, input_size, device)

    # Denormalize data before saving
    generated_data = denormalize_data(generated_data, original_min, original_max)

    # Save output to CSV
    save_to_csv(generated_data)

