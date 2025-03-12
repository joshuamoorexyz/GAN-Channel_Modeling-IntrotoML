import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data (adjust the filename if needed)
data = pd.read_csv('../dataset/2023-12-15_16_53-results.csv')
# Ensure time is sorted
data = data.sort_values(by='time')

# Function to handle path_loss entries
# If the entry is a string representing an array, convert and take the mean
def process_path_loss(value):
    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        try:
            array_values = np.fromstring(value.strip('[]'), sep=' ')
            return np.mean(array_values)
        except ValueError:
            return np.nan
    return value

# Apply the function to the 'path_loss' column
data['path_loss'] = data['path_loss'].apply(process_path_loss)

# Convert 'path_loss' to numeric, forcing errors to NaN
data['path_loss'] = pd.to_numeric(data['path_loss'], errors='coerce')

# Drop rows where 'path_loss' or 'dist' is NaN
data = data.dropna(subset=['path_loss', 'dist'])

# Create a pivot table to reshape for heatmap
# X-axis: time, Y-axis: distance, Values: path_loss
heatmap_data = data.pivot_table(index='dist', columns='time', values='path_loss')

# Fill any remaining missing values (optional)
heatmap_data = heatmap_data.ffill(axis=1).bfill(axis=1)

# Plot the heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(heatmap_data, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False, cbar_kws={'label': 'Path Loss (dB)'})
plt.axis('off')
# plt.title('Path Loss Heatmap Over Time and Distance')
# plt.xlabel('Time Index')
# plt.ylabel('Distance (m)')

# Save the heatmap as an image
plt.savefig('2023-12-15_16_53.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()

