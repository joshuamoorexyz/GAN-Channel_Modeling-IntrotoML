import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load Data from CSV
data = pd.read_csv('../dataset/2023-12-15_15_41-results.csv')

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Distance on X-axis, angle on Y-axis, and SNR on Z-axis
ax.plot(data['dist'], data['aod_theta'], data['avgSnr'], label='aod θ', color='red')
ax.plot(data['dist'], data['aoa_theta'], data['avgSnr'], label='aoa θ', color='blue')

ax.set_xlabel('Distance (meters)')
ax.set_ylabel('Angle (radians)')
ax.set_zlabel('SNR (dB)')
ax.set_title('3D Plot for Angle-Distance-SNR')
ax.legend()

plt.tight_layout()
plt.show()