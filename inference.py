import torch
import pandas as pd
import numpy as np

# Define LSTM-Based Generator (matching training architecture)
class Generator(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=128, num_layers=2):
        super(Generator, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_size)
        self.tanh = torch.nn.Tanh()

    def forward(self, z):
        lstm_out, _ = self.lstm(z)
        output = self.fc(lstm_out)
        return self.tanh(output)

# Load the trained generator model
def load_generator(model_path, input_size, output_size, device="cpu"):
    generator = Generator(input_size, output_size)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.to(device)
    generator.eval()
    return generator

# Denormalization function (assuming min-max scaling during training)
def denormalize_data(generated_data, original_min, original_max):
    return (generated_data + 1) / 2 * (original_max - original_min) + original_min

# Generate synthetic time-series data
def generate_data(generator, num_samples=100, sequence_length=50, input_size=21, device="cpu"):
    noise = torch.randn(num_samples, sequence_length, input_size, device=device)  # Now using LSTM input format
    with torch.no_grad():
        generated_data = generator(noise).cpu().numpy()
    return generated_data

# Save generated data to CSV with feature names
def save_to_csv(data, filename="generated_data.csv"):
    features = ['time', 'center_freq', 'dist', 'h_dist', 'v_dist', 'avgPower', 'avgSnr',
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
    num_samples = 100
    sequence_length = 50  # Must match training
    input_size = 21
    output_size = 21
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load original min/max values used during training
    original_min = np.load("original_min.npy")  
    original_max = np.load("original_max.npy")  

    # Load trained generator
    generator = load_generator(model_path, input_size, output_size, device)

    # Generate synthetic UAV flight data
    generated_data = generate_data(generator, num_samples, sequence_length, input_size, device)

    # Denormalize data before saving
    generated_data = denormalize_data(generated_data, original_min, original_max)

    # Save output to CSV
    save_to_csv(generated_data)

