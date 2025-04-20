import matplotlib.pyplot as plt
import pandas as pd
import os

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Path to the 'dataset' folder
dataset_directory = os.path.join(current_directory, 'dataset')

# List all CSV files in the 'dataset' folder
csv_files = [f for f in os.listdir(dataset_directory) if f.endswith('.csv')]

# Create a plot for each file
for i, csv_file in enumerate(csv_files):
    # Full path to the CSV file
    file_path = os.path.join(dataset_directory, csv_file)
    
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Check if 'avgSnr' column exists and select it
    if 'avgSnr' in data.columns:
        avgSnr_data = data['avgSnr'].values.tolist()  # Extract 'avgSnr' values
    else:
        print(f"Warning: 'avgSnr' column not found in {csv_file}. Skipping this file.")
        continue
    
    # Create an index for the x-axis
    index = list(range(len(avgSnr_data)))
    
    # Create a new figure for each file
    plt.figure(figsize=(10, 6))
    
    # Plot the 'avgSnr' data for the current file
    plt.plot(index, avgSnr_data)
    plt.xlabel('Index')
    plt.ylabel('AvgSnr')
    plt.title(f'AvgSnr Plot of {csv_file}')
    plt.grid(True)
    
    # Show the plot for the current file
    plt.show()
