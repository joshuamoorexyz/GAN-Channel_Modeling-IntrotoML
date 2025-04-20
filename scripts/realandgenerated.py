import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- Configuration ---
FEATURE_COLUMN = 'avg_pl' # The column name we expect in ALL files
GENERATED_DATA_FILE = F"./generated_{FEATURE_COLUMN}_wgan_1000_samples.csv"
REAL_DATA_DIR = './dataset'


# --- Prepare Plot ---
fig, ax = plt.subplots(figsize=(14, 8)) # Create ONE figure and ONE axes object

# --- Generate Colors ---
# Determine how many files we expect to plot to get enough colors
num_real_files = 0
if os.path.isdir(REAL_DATA_DIR):
    # Count only .csv files
    num_real_files = len([f for f in os.listdir(REAL_DATA_DIR) if f.endswith('.csv')])
total_plots_expected = 1 + num_real_files # 1 generated + N real
# Use a colormap that provides distinct colors. Get enough for all expected plots + a buffer.
colors = plt.cm.tab10(np.linspace(0, 1, max(10, total_plots_expected)))

# --- Plot Generated Data ---
print("-" * 20)
print(f"Attempting to plot GENERATED data:")
print(f"  File: {GENERATED_DATA_FILE}")

generated_plotted_successfully = False # Flag to track success

# 1. Check if the generated file exists
if not os.path.exists(GENERATED_DATA_FILE):
    print(f"*** ERROR: Generated file NOT FOUND at '{GENERATED_DATA_FILE}'. Cannot plot it.")
else:
    print(f"  Generated file found. Attempting to read...")
    try:
        # 2. Read the CSV - Assuming header exists (as per corrected inference script)
        gen_data = pd.read_csv(GENERATED_DATA_FILE)

        # 3. Check if data frame is empty
        if gen_data.empty:
             print(f"*** WARNING: Generated file '{GENERATED_DATA_FILE}' is empty.")
        # 4. Check if the expected feature column exists
        elif FEATURE_COLUMN not in gen_data.columns:
            print(f"*** WARNING: Column '{FEATURE_COLUMN}' NOT FOUND in generated file '{GENERATED_DATA_FILE}'.")
            print(f"    Available columns: {list(gen_data.columns)}")
            print(f"    (If no header exists, pd.read_csv reads the first data row as header)")
        else:
            # 5. Extract data and plot
            print(f"  Column '{FEATURE_COLUMN}' found. Plotting...")
            gen_data_points = gen_data[FEATURE_COLUMN].values.tolist()
            gen_index = list(range(len(gen_data_points)))

            # Use the FIRST color (index 0) for generated data, make it distinct
            ax.plot(gen_index, gen_data_points,
                    label='Generated Data',        # Label for legend
                    color=colors[0],               # Use first color
                    linewidth=2.5,                 # Make it thicker
                    zorder=total_plots_expected+1) # Ensure it's drawn on top

            print(f"  Successfully added GENERATED data to the plot.")
            generated_plotted_successfully = True

    # Catch potential errors during reading
    except pd.errors.EmptyDataError:
         print(f"*** ERROR: The file '{GENERATED_DATA_FILE}' seems to be empty or badly formatted (pandas couldn't parse).")
    except Exception as e:
        print(f"*** ERROR: An unexpected error occurred reading/plotting generated file {GENERATED_DATA_FILE}:")
        print(f"    {e}")
print("-" * 20)


# --- Plot Real Data ---
print(f"\nAttempting to plot REAL data:")
print(f"  Directory: {REAL_DATA_DIR}")

# 1. Check if directory exists
if not os.path.isdir(REAL_DATA_DIR):
    print(f"*** ERROR: Real data directory NOT FOUND at '{REAL_DATA_DIR}'. Cannot plot real data.")
else:
    # 2. Get sorted list of CSV files
    # Sorting ensures consistent color assignment if script is re-run
    csv_files = sorted([f for f in os.listdir(REAL_DATA_DIR) if f.endswith('.csv')])

    if not csv_files:
        print(f"  No CSV files found in '{REAL_DATA_DIR}'.")
    else:
        print(f"  Found {len(csv_files)} real data CSV files. Looping through them...")

        # Assign colors starting AFTER the generated data's color
        color_start_index_real = 1

        for i, csv_file in enumerate(csv_files):
            file_path = os.path.join(REAL_DATA_DIR, csv_file)
            current_color_index = color_start_index_real + i
            print(f"   --> Processing: {csv_file} (using color index {current_color_index})")

            try:
                # 3. Read the current real data file
                real_data = pd.read_csv(file_path)

                # 4. Check if empty or column missing
                if real_data.empty:
                     print(f"   *** WARNING: Real data file '{csv_file}' is empty. Skipping.")
                     continue # Skip to next file
                if FEATURE_COLUMN not in real_data.columns:
                    print(f"   *** WARNING: Column '{FEATURE_COLUMN}' NOT FOUND in real file '{csv_file}'. Skipping.")
                    print(f"       Available columns: {list(real_data.columns)}")
                    continue # Skip to next file

                # 5. Extract data and plot onto the SAME axes 'ax'
                real_data_points = real_data[FEATURE_COLUMN].values.tolist()
                real_index = list(range(len(real_data_points)))

                # Check if we have enough predefined colors, fallback or wrap around if not
                if current_color_index >= len(colors):
                     print(f"   *** WARNING: Not enough predefined distinct colors! Reusing color index {current_color_index % len(colors)}.")
                     plot_color = colors[current_color_index % len(colors)]
                else:
                     plot_color = colors[current_color_index]

                ax.plot(real_index, real_data_points,
                        label=csv_file,        # Use filename for legend label
                        color=plot_color,      # Assign specific color
                        alpha=0.8)             # Slight transparency for overlap

                print(f"   Successfully added REAL data from '{csv_file}' to the plot.")

            except pd.errors.EmptyDataError:
                 print(f"   *** ERROR: File '{csv_file}' seems empty or badly formatted. Skipping.")
            except Exception as e:
                print(f"   *** ERROR: An unexpected error occurred reading/plotting real file {csv_file}:")
                print(f"       {e}")
print("-" * 20)


# --- Finalize and Show Plot ---
print("\nFinalizing plot (title, labels, grid, legend)...")
ax.set_xlabel('Index (Time Step)')
ax.set_ylabel(FEATURE_COLUMN)
ax.set_title(f'{FEATURE_COLUMN}: Generated vs. Real Data')
ax.grid(True, linestyle='--', alpha=0.6)

# Only show legend if at least one plot was potentially added
if generated_plotted_successfully or (os.path.isdir(REAL_DATA_DIR) and len(csv_files) > 0):
    ax.legend(loc='best', fontsize='small')
else:
    print("NOTE: No data was successfully plotted, skipping legend.")

plt.tight_layout() # Adjust layout

print("Displaying plot window...")
plt.show()
print("Plot window closed.")