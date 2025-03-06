import os
import h5py
import json
import numpy as np

# Define the directory path containing HDF5 files
date = '2025_02_20'
directory_path = os.path.join(date, 'QuadScan2')
directory_path = os.getcwd()
# Initialize lists to store mean and std arrays
all_Sx_means = []
all_Sx_stds = []
all_Sy_means = []
all_Sy_stds = []
quad_currents = []

# Function to calculate statistics for each HDF5 file
def calculate_statistics(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        images_dataset = h5_file['images']

        Sx = np.array(images_dataset.attrs['Sx'])
        Sy = np.array(images_dataset.attrs['Sy'])

        return Sx, Sy

# Process each HDF5 file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.h5'):
        file_path = os.path.join(directory_path, filename)

        # Extract quad_gradient from the filename
        quad_current = filename.split('_')[2]
        quad_currents.append(quad_current)

        Sx, Sy = calculate_statistics(file_path)

        Sx_mean = np.nanmean(Sx)
        Sx_std = np.nanstd(Sx)
        Sy_mean = np.nanmean(Sy)
        Sy_std = np.nanstd(Sy)

        # Filter out NaN values
        if not np.isnan(Sx_mean):
            all_Sx_means.append(float(Sx_mean))
        if not np.isnan(Sx_std):
            all_Sx_stds.append(float(Sx_std))
        if not np.isnan(Sy_mean):
            all_Sy_means.append(float(Sy_mean))
        if not np.isnan(Sy_std):
            all_Sy_stds.append(float(Sy_std))

# Combine data into a list of tuples
data = list(zip(quad_currents, all_Sx_means, all_Sx_stds, all_Sy_means, all_Sy_stds))

# Sort the data by quad_current
data_sorted = sorted(data, key=lambda x: float(x[0]))

# Unzip the sorted data
quad_currents_sorted, all_Sx_means_sorted, all_Sx_stds_sorted, all_Sy_means_sorted, all_Sy_stds_sorted = zip(*data_sorted)

# Calculate overall statistics
stats = {
    'Sx': {
        'mean': list(all_Sx_means_sorted),
        'std': list(all_Sx_stds_sorted)
    },
    'Sy': {
        'mean': list(all_Sy_means_sorted),
        'std': list(all_Sy_stds_sorted)
    },
    'quad_current': list(quad_currents_sorted)
}

# Save the statistics to a single JSON file
json_file_path = os.path.join(directory_path, 'all_statistics.json')
with open(json_file_path, 'w') as json_file:
    json.dump(stats, json_file, indent=4)

print("Statistics calculated, sorted, and saved in all_statistics.json.")
