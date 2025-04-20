import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
csv_file = './generated_avgSnr_data_1000_samples.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file, header=None)

# Convert the column of data into a list
data_points = data[0].values.tolist()

# Create an index for the x-axis (row numbers)
index = list(range(len(data_points)))

# Plot the data
plt.plot(index, data_points)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('CSV Data Plot')
plt.grid(True)

# Show the plot
plt.show()
