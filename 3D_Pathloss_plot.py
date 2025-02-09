import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geopy.distance import geodesic
import matplotlib.dates as mdates
from datetime import timedelta
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Load Data from CSV
data = pd.read_csv('dataset/2023-12-15_15_41-results.csv')

# Convert 'time' column to numeric values (if not already)
data['time'] = pd.to_numeric(data['time'], errors='coerce')

# Handle missing 'time' entries by dropping rows with NaN values in 'time'
data = data.dropna(subset=['time'])

# Remove rows where all values are NaN
data = data.dropna(how='all')

# Convert lat, lon, alt to numeric values (handle invalid data)
data['lat'] = pd.to_numeric(data['lat'], errors='coerce')
data['lon'] = pd.to_numeric(data['lon'], errors='coerce')
data['alt'] = pd.to_numeric(data['alt'], errors='coerce')

# Handle missing 'lat', 'lon', 'alt' entries by dropping rows with NaN values
data = data.dropna(subset=['lat', 'lon', 'alt'])

# Compute Distance from Starting Point
start_point = (data['lat'].iloc[0], data['lon'].iloc[0])
data['distance_from_start'] = data.apply(lambda row: geodesic(start_point, (row['lat'], row['lon'])).meters, axis=1)

# Convert 'datetime' to numeric for plotting
start_time = data['time'].iloc[0]
data['datetime'] = pd.to_datetime(start_time, unit='s') + data['time'].apply(lambda x: timedelta(seconds=x - start_time))
data['time_numeric'] = mdates.date2num(data['datetime'])

# Clean 'path_loss' column (remove brackets and split)
data['path_loss'] = data['path_loss'].apply(lambda x: np.mean(np.array(x.strip('[]').split(), dtype=float)) if isinstance(x, str) else pd.to_numeric(x, errors='coerce'))

# Drop rows with NaN or Inf in 'path_loss'
data = data.dropna(subset=['path_loss'])

# Normalize the path loss values to fit within a color range (e.g., 0 to 1)
norm = mcolors.Normalize(vmin=data['path_loss'].min(), vmax=data['path_loss'].max())

# Use a colormap to map path loss to colors
cmap = cm.viridis
colors = cmap(norm(data['path_loss']))

# Check for missing or invalid values (NaN or Inf) in key columns
invalid_data = data[['distance_from_start', 'time_numeric', 'alt', 'path_loss']].isna() | data[['distance_from_start', 'time_numeric', 'alt', 'path_loss']].isin([np.inf, -np.inf]).any(axis=1)

if invalid_data.any().any():
    print("Warning: There are invalid values (NaN or Inf) in the data.")
    print(data[invalid_data])

# Drop rows with NaN or Inf values in the critical columns
data = data.dropna(subset=['distance_from_start', 'time_numeric', 'alt', 'path_loss'])

# Check the data ranges before plotting
print("Data ranges:")
print(f"Distance from start: {data['distance_from_start'].min()} to {data['distance_from_start'].max()}")
print(f"Time: {data['time_numeric'].min()} to {data['time_numeric'].max()}")
print(f"Altitude: {data['alt'].min()} to {data['alt'].max()}")

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the cleaned data with path loss color-mapping
scatter = ax.scatter(data['distance_from_start'], data['time_numeric'], data['alt'], c=data['path_loss'], cmap=cmap, marker='o')

# Set Axis Limits
ax.set_xlim(0, data['distance_from_start'].max() + 5)
ax.set_ylim(data['time_numeric'].min(), data['time_numeric'].max())
ax.set_zlim(data['alt'].min() - 1, data['alt'].max() + 1)

# Labels and Formatting
ax.set_xlabel('Distance from Start (meters)')
ax.set_ylabel('Time')
ax.set_zlabel('Altitude (meters)')
ax.set_title('UAV Flight Path: Altitude, Distance, and Path Loss Over Time')
ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

# Highlight Takeoff and Landing Points
ax.scatter(0, data['time_numeric'].iloc[0], data['alt'].iloc[0], color='green', s=100, label='Takeoff')
ax.scatter(0, data['time_numeric'].iloc[-1], data['alt'].iloc[-1], color='red', s=100, label='Landing')

# Add color bar to indicate path loss
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Path Loss (dB)', rotation=270, labelpad=15)

ax.legend()
plt.tight_layout()
plt.show()

