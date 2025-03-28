import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from math import ceil

# Function to handle snr entries
# If the entry is a string representing an array, convert and take the mean
def process_snr(value):
    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        try:
            array_values = np.fromstring(value.strip('[]'), sep=' ')
            return np.mean(array_values)
        except ValueError:
            return np.nan
    return value

#Function for plotting to heatmap
def plot_heatmap(data, save_to, xaxis = 'time', yaxis = 'dist', y_plotrange = [30, 460], values='snr'):

    # Drop rows where values or yaxis are NaN
    print(f"Data before dropping NaN: {data.shape}")
    data = data.dropna(subset=[values, yaxis])
    print(f"Data after dropping NaN: {data.shape}")

    # Create a pivot table to reshape for heatmap
    # X-axis: time, Y-axis: distance, Values: snr
    heatmap_data = data.pivot_table(index=yaxis, columns=xaxis, values=values)

    # Fill any remaining missing values (optional)
    heatmap_data = heatmap_data.ffill(axis=1).bfill(axis=1)

    # Plot the heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(heatmap_data, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False, cbar_kws={'label': 'SNR (dB)'})
    #Force a set range for y axis
    plt.ylim(y_plotrange[0], y_plotrange[1]) 
    #plt.axis('off')

    # Save the heatmap as an image
    plt.savefig(save_to, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


#Start executable

source_dir = "./dataset/reduced_only_full/"
output_dir = "./Figures/SNR-H"
chunk_size = 300

for file in os.listdir(source_dir):
    print(f"Found: {file}")
    if file.endswith(".csv"):
        #Load the dataset
        file_path = os.path.join(source_dir, file)
        df = pd.read_csv(file_path)

        #Sorted by time
        df = df.sort_values(by='time')
        df['snr'] = df['snr'].apply(process_snr)
        df['snr'] = pd.to_numeric(df['snr'], errors='coerce')
        #print(df[['snr', 'v_dist', 'h_dist']].describe())

        #Base target for writing for results from this file
        folder_base = file.replace(".csv", "")
        results_target = f"{output_dir}{folder_base}_SNR_vdist"

        #Make the directorie
        os.makedirs(results_target, exist_ok=True)

        #For each chunk
        for i in range(ceil(len(df) / chunk_size)):
            start_idx = i * chunk_size
            end_idx =  min(start_idx + chunk_size, len(df))
            df_chunk = df.iloc[start_idx:end_idx]

            #For current data:
            #snr: -19 to -4
            #v_dist: 0 to 100
            #h_dist: 30 to 460

            output_path = os.path.join(results_target, f"Heatmap_{i+1}.png")
            plot_heatmap(df_chunk, output_path, xaxis='time', yaxis='h_dist', y_plotrange = [30, 460], values='snr')

            print(f"Chunk {i+1} visualization saved for {file}!")

print("All visualizations saved!")