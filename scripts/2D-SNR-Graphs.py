import pandas as pd
import matplotlib.pyplot as plt
import os
from math import ceil




#Start executable

source_dir = "./dataset/reduced_only_full/"
output_dir = "./Figures/SNR_v_h_dist/V/"
chunk_size = 20000 #Take everything at the moment, but may want to do it in smaller batchess later

for file in os.listdir(source_dir):
    print(f"Found: {file}")
    if file.endswith(".csv"):
        #Load the dataset
        file_path = os.path.join(source_dir, file)
        df = pd.read_csv(file_path)

        #Sorted by time
        df = df.sort_values(by='time')
        # df['snr'] = df['snr'].apply(process_snr)
        # df['snr'] = pd.to_numeric(df['snr'], errors='coerce')
        

        #Base target for writing for results from this file
        output_filename = file.replace(".csv", "")
        results_target = f"{output_dir}{output_filename}.png"

        #Make the directory
        os.makedirs(output_dir, exist_ok=True)

        #For each chunk
        for i in range(ceil(len(df) / chunk_size)):
            start_idx = i * chunk_size
            end_idx =  min(start_idx + chunk_size, len(df))
            df_chunk = df.iloc[start_idx:end_idx]

            #For current data:
            #snr: -19 to -4
            #v_dist: 0 to 100
            #h_dist: 30 to 460

            plt.figure(figsize=(10, 10))
            plt.scatter(df_chunk['h_dist'], df_chunk['avgSnr'], alpha=0.5, s=1)
            plt.ylim(-19, -4)
            plt.xlim(right=100)
            print(df_chunk['v_dist'].describe())
            plt.axis('off')
            plt.savefig(results_target, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()

            print(f"Chunk {i+1} visualization saved for {file}!")

print("All visualizations saved!")