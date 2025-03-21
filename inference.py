import torch
import pandas as pd
import numpy as np

# Define the Generator class (matching training architecture)
class Generator(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_size),
            torch.nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Load the trained generator model
def load_generator(model_path, input_size, output_size, device="cpu"):
    generator = Generator(input_size, output_size)  # Instantiate the model with correct dimensions
    generator.load_state_dict(torch.load(model_path, map_location=device))  # Load saved weights
    generator.to(device)
    generator.eval()
    return generator

# Generate synthetic data
def generate_data(generator, num_samples=100, input_size=20, device="cpu"):
    noise = torch.randn(num_samples, input_size, device=device)
    with torch.no_grad():
        generated_data = generator(noise).cpu().numpy()
    return generated_data

# Save generated data to CSV
def save_to_csv(data, filename="generated_data.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved generated data to {filename}")

if __name__ == "__main__":
    model_path = "generatorcheckpoint.pth"  # Path to trained generator model
    num_samples = 100  # Number of samples to generate
    input_size = 20  # Feature count (matches training data)
    output_size = 20  # Output size should match input features
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = load_generator(model_path, input_size, output_size, device)
    generated_data = generate_data(generator, num_samples, input_size, device)
    save_to_csv(generated_data)

