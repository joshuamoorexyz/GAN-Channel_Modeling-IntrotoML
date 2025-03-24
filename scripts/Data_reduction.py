import pandas as pd
import time
import os

#Target directories
source_dir = "./dataset/"
output_dir = "./dataset/reduced_only_full/"

#Make the dir exist
os.makedirs(output_dir, exist_ok=True)

#Time stuff
start_time = time.time()

#Target features
features = ['center_freq', 'dist', 'h_dist', 'v_dist', 'avgPower', 'avgSnr', 'freq_offset', 'avg_pl', 'aod_theta',
            'aoa_theta', 'aoa_phi', 'pitch', 'yaw', 'roll', 'vel_x', 'vel_y', 'vel_z', 'speed', 'avg_pl_rolling', 'avg_pl_ewma']

#For all .csv files in sourceDir
for file in os.listdir(source_dir):
    print(f"found: {file}")
    if file.endswith(".csv") and "results" in file:
        #Figure out the pathing
        pathing = os.path.join(source_dir, file)
        df = pd.read_csv(pathing)

        #used_freq = center_freq + freq_offset (Don't want both. Either use summation or the frequency offset) Using summation here
        df['used_freq'] = df['center_freq'] + df['freq_offset']

        #Remove old columns and add used_freq
        final_features = [col for col in features if col not in ['center_freq', 'freq_offset']] + ['used_freq']
        df_reduced = df[final_features]

        #Purge if missing
        df_reduced = df_reduced.dropna()

        #Fill with 0s
        # df_reduced = df_reduced.fillna(0)

        #Write the modified dataframe to a new CSV file
        outputPath = os.path.join(output_dir, file)
        df_reduced.to_csv(outputPath, index=False) #Don't need numbering for rows; time is a primary key

        print(f"Reduced {pathing} and stored to {outputPath}")
    else:
        print(f"Skipping {file}")

print(f"Finished processing in {time.time() - start_time} s")