import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Importing required variables and generator class to avoid misconfiguration
from train import NOISE_DIM, OUTPUT_DIM, LSTM_HIDDEN_G, LSTM_LAYERS_G, MIN_MAX_NPZ_PATH, TARGET_FEATURES
from train import SEQUENCE_LENGTH as TRAINING_SEQUENCE_LENGTH, GEN_CHECKPOINT_FINAL_PATH as MODEL_PATH
from generator import Generator

OUTPUT_CSV_FILENAME = f"generated_{TARGET_FEATURES}_wgan_1000_samples.csv" # Output filename

# Generate exactly 1 sequence of length 1000 to get 1000 total samples
NUM_SEQUENCES_TO_GENERATE = 1
# Total data points = NUM_SEQUENCES_TO_GENERATE * TRAINING_SEQUENCE_LENGTH = 1 * 1000 = 1000

# Function to load the trained generator model
def load_generator(model_path, noise_dim, seq_len, output_dim, lstm_hidden, lstm_layers, device="cpu"):
    # Instantiate the generator with the parameters used during training
    generator = Generator(noise_dim, seq_len, output_dim, lstm_hidden, lstm_layers).to(DEVICE)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        # Load the state dictionary
        generator.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Successfully loaded model weights from {model_path}")
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        print("Ensure the Generator class definition exactly matches the one used for training.")
        raise
    generator.eval() # Set model to evaluation mode
    return generator

# Denormalization function (Handles Tanh output [-1, 1] and original normalization [0, 1])
def denormalize_data(generated_data_normalized, global_min, global_max):
    # Input generated_data_normalized is in range [-1, 1] due to Tanh
    # Step 1: Map [-1, 1] to [0, 1] (the range of the normalized training data)
    generated_data_01 = (generated_data_normalized + 1) / 2
    # Step 2: Map [0, 1] back to original scale [global_min, global_max]
    # Ensure global_min/max have the same dimension as the feature dimension for broadcasting
    min_val = global_min.reshape(1, 1, -1) # Reshape for broadcasting (1, 1, num_features)
    max_val = global_max.reshape(1, 1, -1) # Reshape for broadcasting (1, 1, num_features)
    denormalized_data = generated_data_01 * (max_val - min_val) + min_val
    return denormalized_data

# Generate synthetic time-series data
def generate_data(generator, num_sequences, sequence_length, noise_dim, device="cpu"):
    # Noise shape: (num_sequences, sequence_length, noise_dim)
    noise = torch.randn(num_sequences, sequence_length, noise_dim, device=DEVICE)
    print(f"Generated noise shape: {noise.shape}")
    with torch.no_grad(): # Essential for inference
        generated_data = generator(noise)
    print(f"Raw generator output shape (normalized -1 to 1): {generated_data.shape}")
    return generated_data.cpu().numpy() # Move data to CPU and convert to numpy

# Save generated data to CSV with feature names
def save_to_csv(data, filename="generated_data.csv", features=[TARGET_FEATURES]):
    if data.ndim != 3:
        print(f"Error: Expected data with 3 dimensions (sequences, seq_len, features), but got {data.ndim} dimensions.")
        return
    num_sequences, sequence_length, num_features = data.shape
    print(f"Data shape before reshape for CSV: {data.shape}")

    # Reshape the data: Stack sequences vertically -> (num_sequences * sequence_length, num_features)
    reshaped_data = data.reshape(-1, num_features)
    print(f"Data shape after reshape for CSV: {reshaped_data.shape}")

    if reshaped_data.shape[1] != len(features):
         print(f"Warning: Number of features in data ({reshaped_data.shape[1]}) does not match number of feature names ({len(features)}). Using provided feature names anyway.")

    df = pd.DataFrame(reshaped_data, columns=features)
    df.to_csv(filename, index=False)
    print(f"Saved {reshaped_data.shape[0]} data points to {filename}")

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- Load combined global min/max from .npz file ---
    global_min = None
    global_max = None
    try:
        if not os.path.exists(MIN_MAX_NPZ_PATH):
            raise FileNotFoundError(f"Min/max NPZ file not found: {MIN_MAX_NPZ_PATH}")

        min_max_data = np.load(MIN_MAX_NPZ_PATH)
        # Extract arrays using the keys used during saving in train.py
        global_min = min_max_data[TARGET_FEATURES +"_min"]
        global_max = min_max_data[TARGET_FEATURES +"_max"]

        # Ensure min/max are numpy arrays (they should be from np.load)
        global_min = np.array(global_min)
        global_max = np.array(global_max)

        print(f"\nLoaded global min/max from {MIN_MAX_NPZ_PATH}")
        print(f"Global min shape: {global_min.shape}, Global max shape: {global_max.shape}")
        print(f"Global min values: {global_min}")
        print(f"Global max values: {global_max}")

        # Validation check
        if global_min.shape != global_max.shape:
             raise ValueError(f"Min shape {global_min.shape} and Max shape {global_max.shape} do not match.")
        if np.any(global_max <= global_min):
            print("Warning: At least one global max value <= corresponding global min value. Check min/max calculation.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure the WGAN-GP training script ran successfully and saved the .npz file.")
        exit()
    except KeyError as e:
         print(f"Error: Key {e} not found in {MIN_MAX_NPZ_PATH}. Expected 'global_min' and 'global_max'.")
         exit()
    except Exception as e:
        print(f"An error occurred loading the min/max NPZ file: {e}")
        exit()
    # --- End loading min/max ---

    # Load the trained WGAN-GP generator
    print(f"\nLoading generator from {MODEL_PATH}...")
    try:
        generator = load_generator(
            model_path=MODEL_PATH,
            noise_dim=NOISE_DIM,
            seq_len=TRAINING_SEQUENCE_LENGTH,
            output_dim=OUTPUT_DIM,
            lstm_hidden=LSTM_HIDDEN_G,
            lstm_layers=LSTM_LAYERS_G,
            device=DEVICE
        )
        print("Generator loaded successfully.")
    except Exception as e:
        print(f"Failed to load generator: {e}")
        exit()

    # Generate synthetic data sequences
    print(f"\nGenerating {NUM_SEQUENCES_TO_GENERATE} sequence(s) of length {TRAINING_SEQUENCE_LENGTH}...")
    generated_normalized_data = generate_data(
        generator=generator,
        num_sequences=NUM_SEQUENCES_TO_GENERATE,
        sequence_length=TRAINING_SEQUENCE_LENGTH,
        noise_dim=NOISE_DIM,
        device=DEVICE
    )
    # Optional: Print stats of the raw normalized [-1, 1] output
    print(f"Normalized data stats (raw output): Min={np.min(generated_normalized_data):.4f}, Max={np.max(generated_normalized_data):.4f}, Mean={np.mean(generated_normalized_data):.4f}")

    # Denormalize data before saving
    print("\nDenormalizing data...")
    try:
        generated_denormalized_data = denormalize_data(generated_normalized_data, global_min, global_max)
        print(f"Denormalized data shape: {generated_denormalized_data.shape}")
         # Optional: print some stats of the denormalized data
        print(f"Denormalized data stats (final output): Min={np.min(generated_denormalized_data):.4f}, Max={np.max(generated_denormalized_data):.4f}, Mean={np.mean(generated_denormalized_data):.4f}")

    except Exception as e:
         print(f"An error occurred during denormalization: {e}")
         print(f"  Normalized data shape: {generated_normalized_data.shape}")
         print(f"  Global min shape: {global_min.shape}, Global max shape: {global_max.shape}")
         exit()

    # Save output to CSV
    print("\nSaving data to CSV...")
    save_to_csv(generated_denormalized_data, filename=OUTPUT_CSV_FILENAME, features=[TARGET_FEATURES])

    print("\nInference complete.")
