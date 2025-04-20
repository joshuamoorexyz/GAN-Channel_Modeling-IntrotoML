# --- Imports ---
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.autograd as autograd
from discriminator import Critic
from generator import Generator, LSTM_HIDDEN_G, LSTM_LAYERS_G


# --- Parser for command-line ---
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='The learning rate to use for training',
    )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=402,
        help='The number of epochs to train the model',
    )
    

    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        for arg in unknown:
            print(f'Unknown argument: {arg}.\nCheck --help for reference.')
            sys.exit(1)
    
    if args.lr <= 0:
        parser.error('--lr must be greater than 0')
        
    if args.num_epochs < 1:
        parser.error('--num_epochs must be greater than or equal to 1')
    
    return args
  

# --- Global variables ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_FEATURES = ['avgPower', 'avgSnr', 'freq_offset', 'avg_pl', 'aod_theta', 'speed'] # What we want to generate
CONDITION_FEATURES = ['dist', 'alt', 'h_dist', 'v_dist', 'vel_x', 'vel_y', 'vel_z']     # Inputs influencing the target
ALL_FEATURES = TARGET_FEATURES + CONDITION_FEATURES # All columns needed from CSV

SEQUENCE_LENGTH = 1000 # Adjust based on expected flight duration / signal patterns
BATCH_SIZE = 32

NOISE_DIM = 10 # Dimension of the random noise vector z (can be tuned)
OUTPUT_DIM = len(TARGET_FEATURES) # Generator output dimension (just avg_pl)
CONDITION_DIM = len(CONDITION_FEATURES)
GENERATOR_INPUT_DIM = NOISE_DIM + CONDITION_DIM

# Evaluation and Saving
EVAL_FREQ = 100
CHECKPOINT_FREQ = 200
SAMPLE_DIR = "generated_samples_conditional/" 
CHECKPOINT_DIR = "checkpoints_conditional/" 
MIN_MAX_NPZ_PATH = "original_min_max_conditional.npz"
GEN_CHECKPOINT_FINAL_PATH = "generator_wgan_gp_conditional_final.pth" 
CRITIC_CHECKPOINT_FINAL_PATH = "critic_wgan_gp_conditional_final.pth" 


def main() -> None:
    
    args = parse_arguments()
     
    TRAIN_DIR = "dataset/train"

    # WGAN-GP Hyperparameters
    LEARNING_RATE = args.lr  # Default is 1e-4
    B1 = 0.0
    B2 = 0.9
    N_CRITIC = 5
    LAMBDA_GP = 10
    EPOCHS = args.num_epochs # Default is 402; adjust based on observation

    # --- Create Output Directories ---
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Print Configuration ---
    print("--- Conditional WGAN-GP Configuration ---")
    print(f"Using device: {DEVICE}")
    print(f"Target Feature(s): {TARGET_FEATURES}")
    print(f"Condition Feature(s): {CONDITION_FEATURES}")
    print(f"Sequence Length: {SEQUENCE_LENGTH}")
    print(f"LSTM Hidden (G): {LSTM_HIDDEN_G}")
    print(f"Noise Dim (z): {NOISE_DIM}")
    print(f"Condition Dim: {CONDITION_DIM}")
    print(f"Generator Input Dim: {GENERATOR_INPUT_DIM}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"n_critic: {N_CRITIC}")
    print(f"Lambda GP: {LAMBDA_GP}")
    print("----------------------------------------")

    # --- Load and Prepare Data ---
    def load_multiple_csvs(directory, features_to_load):
        dataframes = []
        print(f"Loading data from: {directory}")
        try:
            files = [f for f in os.listdir(directory) if f.endswith(".csv")]
            if not files:
                raise FileNotFoundError(f"No CSV files found in directory: {directory}")
            print(f"Found {len(files)} CSV files. Loading columns: {features_to_load}")
            for file in sorted(files):
                file_path = os.path.join(directory, file)
                try:
                    if os.path.getsize(file_path) > 0:
                        # Load only the required columns
                        df_temp = pd.read_csv(file_path, usecols=features_to_load)
                        if not df_temp.empty:
                            # Check if all specified features exist after loading
                            if all(feat in df_temp.columns for feat in features_to_load):
                                dataframes.append(df_temp[features_to_load]) # Ensure correct column order
                            else:
                                missing = [f for f in features_to_load if f not in df_temp.columns]
                                print(f"Warning: Skipping {file} because it's missing required columns: {missing}")
                        else:
                            print(f"Warning: Skipping empty dataframe from file {file}")
                    else:
                        print(f"Warning: Skipping empty file {file}")
                except ValueError as e:
                    print(f"Warning: Error reading columns from {file}. Error: {e}. Skipping.")
                except Exception as e:
                    print(f"Error reading {file}: {e}. Skipping.")

            if not dataframes:
                raise ValueError("No valid data loaded. Ensure CSVs exist, have the correct columns, and are not empty.")

            combined_df = pd.concat(dataframes, ignore_index=True)
            print(f"Combined dataframe shape: {combined_df.shape}")
            return combined_df

        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during data loading: {e}")
            raise

    # Load dataset with ALL necessary features
    df_raw = load_multiple_csvs(TRAIN_DIR, ALL_FEATURES)
    df_raw.fillna(0, inplace=True) # Handle NaNs

    # --- Calculate and Save Global Min/Max for ALL features --- ### CHANGE ###
    all_mins = {}
    all_maxs = {}
    print("\nCalculating min/max for all features...")
    for col in ALL_FEATURES:
        min_val = df_raw[col].min()
        max_val = df_raw[col].max()
        all_mins[col + "_min"] = min_val # Store with specific key
        all_maxs[col + "_max"] = max_val # Store with specific key
        if min_val == max_val:
            print(f"Warning: Feature '{col}' has min == max ({min_val}). Adding epsilon to max for normalization.")
            all_maxs[col + "_max"] += 1e-8
        print(f"  {col}: min={min_val:.4f}, max={max_val:.4f}")

    np.savez(MIN_MAX_NPZ_PATH, **all_mins, **all_maxs) # Save all min/max values
    print(f"Saved global min/max values for {ALL_FEATURES} to {MIN_MAX_NPZ_PATH}")


    # --- Normalize ALL data to [-1, 1] range --- ### CHANGE ###
    def normalize_data_neg1_pos1(df_in, features_list, glob_min_max_dict):
        df_norm = df_in.copy()
        for col in features_list:
            g_min = glob_min_max_dict[col + "_min"]
            g_max = glob_min_max_dict[col + "_max"]
            # Scale to [0, 1] first
            df_norm[col] = (df_norm[col] - g_min) / (g_max - g_min + 1e-8)
            # Then scale to [-1, 1]
            df_norm[col] = (df_norm[col] * 2) - 1
        return df_norm

    df_normalized = normalize_data_neg1_pos1(df_raw, ALL_FEATURES, {**all_mins, **all_maxs})
    print("\nAll required features normalized to [-1, 1] range.")

    # --- Denormalize function (for evaluation, only needs target usually) --- ### CHANGE ###
    def denormalize(data_tensor_or_array, feature_name, glob_min_max_dict):
        """Denormalizes data for a specific feature from [-1, 1] range back to original scale."""
        if feature_name + "_min" not in glob_min_max_dict or feature_name + "_max" not in glob_min_max_dict:
            raise KeyError(f"Min/Max values for feature '{feature_name}' not found in provided dict.")

        g_min = glob_min_max_dict[feature_name + "_min"]
        g_max = glob_min_max_dict[feature_name + "_max"]

        if isinstance(data_tensor_or_array, torch.Tensor):
            data = data_tensor_or_array.detach().cpu().numpy()
        else:
            data = data_tensor_or_array

        data_01 = (data + 1) / 2
        denormalized_data = data_01 * (g_max - g_min) + g_min
        return denormalized_data


    # --- Convert data into sequences --- ### CHANGE ###
    print(f"\nCreating sequences of length {SEQUENCE_LENGTH}...")
    target_sequences = []
    condition_sequences = []
    if len(df_normalized) <= SEQUENCE_LENGTH:
        raise ValueError(f"Dataset length ({len(df_normalized)}) insufficient for sequence length ({SEQUENCE_LENGTH}).")

    # Extract numpy arrays for faster slicing
    target_data_np = df_normalized[TARGET_FEATURES].values
    condition_data_np = df_normalized[CONDITION_FEATURES].values

    for i in range(len(df_normalized) - SEQUENCE_LENGTH + 1):
        target_sequences.append(target_data_np[i : i + SEQUENCE_LENGTH])
        condition_sequences.append(condition_data_np[i : i + SEQUENCE_LENGTH])

    if not target_sequences or not condition_sequences:
        raise ValueError("Failed to create sequences. Check data.")

    # Convert lists of sequences to tensors
    # Target shape: (num_sequences, sequence_length, num_target_features)
    # Condition shape: (num_sequences, sequence_length, num_condition_features)
    target_tensor = torch.tensor(np.array(target_sequences), dtype=torch.float32)
    condition_tensor = torch.tensor(np.array(condition_sequences), dtype=torch.float32)

    print(f"Created {target_tensor.shape[0]} sequences.")
    print(f"Target tensor shape: {target_tensor.shape}")
    print(f"Condition tensor shape: {condition_tensor.shape}")

    # Create dataset yielding (condition, target) tuples
    dataset = TensorDataset(condition_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


    ############################
    # Gradient Penalty Function
    ############################
    # Calculates GP based ONLY on the target sequences (e.g., avgSnr)
    def compute_gradient_penalty(critic, real_target_samples, real_condition_samples, fake_target_samples):
        """Calculates the gradient penalty loss for WGAN GP on target samples"""
        batch_size = real_target_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, device=DEVICE)
        alpha = alpha.expand_as(real_target_samples) # Expand to target sample shape

        # Interpolate only the target samples
        interpolates_target = (alpha * real_target_samples + ((1 - alpha) * fake_target_samples)).requires_grad_(True)
        d_interpolates_target = critic(interpolates_target, real_condition_samples) # Critic takes only target

        fake_outputs = torch.ones(d_interpolates_target.size(), device=DEVICE, requires_grad=False)
        gradients = autograd.grad(
            outputs=d_interpolates_target,
            inputs=interpolates_target, # Gradients w.r.t. interpolated target
            grad_outputs=fake_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.reshape(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    ############################
    # WGAN-GP Training Function
    ############################
    def train_wgan_gp(generator, critic, dataloader, epochs=EPOCHS, n_critic=N_CRITIC, lambda_gp=LAMBDA_GP):
        # Load all min/max values for denormalization during evaluation
        try:
            min_max_dict = dict(np.load(MIN_MAX_NPZ_PATH)) # Load into a dictionary
            print("Loaded min/max for denormalization during evaluation.")
        except Exception as e:
            print(f"Error loading min/max file {MIN_MAX_NPZ_PATH}: {e}")
            min_max_dict = None # Proceed without evaluation plotting if file fails

        optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(B1, B2))
        optimizer_C = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(B1, B2))

        gen_losses = []
        critic_losses = []
        wasserstein_distances = []

        print("\n--- Starting Conditional WGAN-GP Training ---")
        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_c_loss = 0.0
            epoch_wd = 0.0
            critic_updates_this_epoch = 0
            gen_updates_this_epoch = 0

            # Dataloader yields (condition_batch, target_batch)
            for i, (real_condition_seq, real_target_seq) in enumerate(dataloader):
                real_condition_seq = real_condition_seq.to(DEVICE)
                real_target_seq = real_target_seq.to(DEVICE) # e.g., avgSnr
                current_batch_size = real_target_seq.size(0)

                # --------------
                #  Train Critic
                # --------------
                optimizer_C.zero_grad()

                # Generate noise z
                z = torch.randn(current_batch_size, SEQUENCE_LENGTH, NOISE_DIM, device=DEVICE)
                # Combine noise and the REAL condition sequence as input to Generator
                gen_input = torch.cat((z, real_condition_seq), dim=2)

                # Generate fake TARGET sequence (e.g., fake avgSnr)
                fake_target_seq = generator(gen_input).detach()

                # Critic evaluates REAL TARGET sequence
                real_output = critic(real_target_seq, real_condition_seq)
                # Critic evaluates FAKE TARGET sequence
                fake_output = critic(fake_target_seq, real_condition_seq)

                # Calculate gradient penalty using only TARGET sequences
                gradient_penalty = compute_gradient_penalty(critic, real_target_seq.data, real_condition_seq.data, fake_target_seq.data)

                # Critic loss: D(fake_target) - D(real_target) + GP
                loss_C = fake_output.mean() - real_output.mean() + lambda_gp * gradient_penalty
                epoch_c_loss += loss_C.item()
                epoch_wd += (real_output.mean() - fake_output.mean()).item() # W-Dist Est based on target

                loss_C.backward()
                optimizer_C.step()
                critic_updates_this_epoch += 1

                # Train the generator only every n_critic iterations
                if i % n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------
                    optimizer_G.zero_grad()

                    # Generate new noise z
                    z_gen = torch.randn(current_batch_size, SEQUENCE_LENGTH, NOISE_DIM, device=DEVICE)
                    # Combine noise and REAL condition sequence
                    gen_input_for_G = torch.cat((z_gen, real_condition_seq), dim=2)
                    # Generate fake TARGET sequence
                    gen_fake_target = generator(gen_input_for_G)

                    # Get critic score for the generated TARGET sequence
                    gen_fake_output = critic(gen_fake_target, real_condition_seq)

                    # Generator loss: maximize D(fake_target) -> minimize -D(fake_target)
                    loss_G = -gen_fake_output.mean()
                    epoch_g_loss += loss_G.item()

                    loss_G.backward()
                    optimizer_G.step()
                    gen_updates_this_epoch += 1

            # --- End of Epoch ---
            avg_epoch_c_loss = epoch_c_loss / critic_updates_this_epoch if critic_updates_this_epoch > 0 else 0
            avg_epoch_g_loss = epoch_g_loss / gen_updates_this_epoch if gen_updates_this_epoch > 0 else 0
            avg_epoch_wd = epoch_wd / critic_updates_this_epoch if critic_updates_this_epoch > 0 else 0

            critic_losses.append(avg_epoch_c_loss)
            gen_losses.append(avg_epoch_g_loss)
            wasserstein_distances.append(avg_epoch_wd)

            print(
                f"[Epoch {epoch+1}/{epochs}] "
                f"[C loss: {avg_epoch_c_loss:.4f}] "
                f"[G loss: {avg_epoch_g_loss:.4f}] "
                f"[W-Dist Est: {avg_epoch_wd:.4f}]"
            )

            # --- Periodic Evaluation ---
            if (epoch % EVAL_FREQ == 0 or epoch == epochs - 1) and min_max_dict is not None:
                generator.eval()
                with torch.no_grad():
                    # Take the first condition sequence from the last batch for plotting demo
                    # In real inference, you'd use your target flight path here
                    sample_condition_seq = real_condition_seq[:1] # Take first sequence (batch size 1)
                    num_samples_to_plot = 1 # Plot only one sample for clarity

                    z_sample = torch.randn(num_samples_to_plot, SEQUENCE_LENGTH, NOISE_DIM, device=DEVICE)
                    sample_gen_input = torch.cat((z_sample, sample_condition_seq), dim=2)
                    fake_target_samples_norm = generator(sample_gen_input) # [-1, 1] range

                    n_cols = min(3, OUTPUT_DIM)  # OUTPUT_DIM = len(target_features)
                    n_rows = (OUTPUT_DIM + n_cols - 1) // n_cols
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
                    for i, target_feature in enumerate(TARGET_FEATURES):

                        # Denormalize the generated TARGET feature (e.g., avgSnr)
                        fake_target_denorm = denormalize(fake_target_samples_norm[:, :, i], # Select the feature column
                                                        target_feature,
                                                        min_max_dict)

                        # Plotting
                        axes[i//3, i%3].plot(fake_target_denorm[0, :], label=f'Generated {target_feature}')
                        axes[i//3, i%3].set_title(f'Generated Sample (Denormalized {target_feature}) - Epoch {epoch+1}')
                        axes[i//3, i%3].set_xlabel('Time Step')
                        axes[i//3, i%3].set_ylabel(target_feature)
                        # Optional: Add y-limits based on original range
                        plot_min = min_max_dict[target_feature + "_min"]
                        plot_max = min_max_dict[target_feature + "_max"]
                        plot_min -= abs(plot_min * 0.1)
                        plot_max += abs(plot_max * 0.1)
                        axes[i//3, i%3].set_ylim(plot_min, plot_max)
                        axes[i//3, i%3].grid(True, linestyle='--', alpha=0.6)
                        axes[i//3, i%3].legend()
                    plt.tight_layout()
                    plot_filename = os.path.join(SAMPLE_DIR, f"generated_sample_epoch_{epoch+1}.png")
                    plt.savefig(plot_filename, dpi=300)
                    plt.close()
                    print(f"Saved generated sample plot: {plot_filename}")

                generator.train()

            # --- Periodic Checkpoint Saving ---
            if epoch % CHECKPOINT_FREQ == 0 or epoch == epochs - 1:
                gen_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch+1}.pth")
                critic_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"critic_epoch_{epoch+1}.pth")
                torch.save(generator.state_dict(), gen_checkpoint_path)
                torch.save(critic.state_dict(), critic_checkpoint_path)
                print(f"Saved checkpoints for epoch {epoch+1} to {CHECKPOINT_DIR}")

        print("--- Training Complete ---")
        return gen_losses, critic_losses, wasserstein_distances

    ###############################
    # Initialize and Train WGAN-GP
    ###############################
    generator = Generator(generator_input_dim=GENERATOR_INPUT_DIM,
                        seq_len=SEQUENCE_LENGTH,
                        output_dim=OUTPUT_DIM).to(DEVICE)

    critic = Critic(target_dim=OUTPUT_DIM, condition_dim=CONDITION_DIM, # Critic input dim is target dim
                    seq_len=SEQUENCE_LENGTH).to(DEVICE)


    # Train the models
    g_losses, c_losses, w_distances = train_wgan_gp(generator, critic, dataloader)

    # Save the final trained models
    torch.save(generator.state_dict(), GEN_CHECKPOINT_FINAL_PATH)
    torch.save(critic.state_dict(), CRITIC_CHECKPOINT_FINAL_PATH)
    print(f"\nSaved FINAL Generator checkpoint to {GEN_CHECKPOINT_FINAL_PATH}")
    print(f"Saved FINAL Critic checkpoint to {CRITIC_CHECKPOINT_FINAL_PATH}")

    # --- Plot Losses ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(c_losses, label="Critic Loss")
    plt.title("Generator and Critic Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(w_distances, label="Wasserstein Distance Estimate")
    plt.title("Estimated Wasserstein Distance (Target)")
    plt.xlabel("Epochs")
    plt.ylabel("Distance Estimate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_filename = "wgan_gp_conditional_loss_plot.png"
    plt.savefig(loss_plot_filename, dpi=300)
    print(f"Saved loss plot as {loss_plot_filename}")
    # plt.show()

    print("\nConditional training script finished.")


if __name__ == '__main__':
    main()
    