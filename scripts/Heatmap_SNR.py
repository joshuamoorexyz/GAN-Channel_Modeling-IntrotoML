import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data (adjust the filename if needed)
data = pd.read_csv('../dataset/2023-12-15_16_53-results.csv')

# Ensure time is sorted
data = data.sort_values(by='time')

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

# Apply the function to the 'snr' column
data['snr'] = data['snr'].apply(process_snr)

# Convert 'snr' to numeric, forcing errors to NaN
data['snr'] = pd.to_numeric(data['snr'], errors='coerce')

# Drop rows where 'snr' or 'dist' is NaN
data = data.dropna(subset=['snr', 'dist'])

# Create a pivot table to reshape for heatmap
# X-axis: time, Y-axis: distance, Values: snr
heatmap_data = data.pivot_table(index='dist', columns='time', values='snr')

# Fill any remaining missing values (optional)
heatmap_data = heatmap_data.ffill(axis=1).bfill(axis=1)

# Plot the heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(heatmap_data, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False, cbar_kws={'label': 'SNR (dB)'})
plt.axis('off')
# plt.title('SNR Heatmap Over Time and Distance')
# plt.xlabel('Time Index')
# plt.ylabel('Distance (m)')

# Save the heatmap as an image
plt.savefig('2023-12-15_16_53.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()

