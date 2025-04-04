import pandas as pd
import time
import os

#Target directories
source_dir = "./dataset/"
output_dir = "./dataset/Moving_Only/"

#Make the dir exist
os.makedirs(output_dir, exist_ok=True)

#Time stuff
start_time = time.time()

#Target features

#For all .csv files in sourceDir
for file in os.listdir(source_dir):
    print(f"found: {file}")
    if file.endswith(".csv") and "results" in file:
        #Figure out the pathing
        pathing = os.path.join(source_dir, file)
        df = pd.read_csv(pathing)

        #Thresholding
        speed_thresh = 0.15 #Detect once drone is actually intending to move
        consecutive_slow = 0
        max_slow = 5

        #Segments - Blame files 5, 8, & 9
        segments = []
        current_start = None
        #start = df[df['speed'] > speed_thresh].index.min()

        #Sequentially parse data until max_slow is reached then chuck it into segments
        for i in range(len(df)):
            if df.loc[i, 'speed'] > speed_thresh:
                if current_start is None:
                    current_start = i
                consecutive_slow = 0
            else:
                consecutive_slow += 1
                if consecutive_slow >= max_slow and current_start is not None:
                    # End of a moving segment
                    segments.append((current_start, i - max_slow))
                    current_start = None
                    consecutive_slow = 0

        #JIC ending on movement
        if current_start is not None:
            segments.append((current_start, len(df) - 1))


        #Select the largest one
        df_reduced = None
        if segments:
            #Find the max segment & turn it into start & end points
            start, stop = max(segments, key=lambda x: x[1] - x[0])
            df_reduced = df.loc[start:stop]
        else:
            df_reduced = df

        #Write the modified dataframe to a new CSV file
        outputPath = os.path.join(output_dir, file)
        df_reduced.to_csv(outputPath, index=False) #Don't need numbering for rows; time is a primary key

        print(f"Reduced {pathing} and stored to {outputPath}")
    else:
        print(f"Skipping {file}")

print(f"Finished processing in {time.time() - start_time} s")