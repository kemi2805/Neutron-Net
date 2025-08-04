# src/data/processing.py
"""
Processing module for neutron star diffusion.
Migrated and cleaned up.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

import os
import dask
import dask.dataframe as dd
import dask.bag as db
from dask import delayed

from dask.distributed import Client
import time

def process_file(file_path):
    try:
        df = pd.read_parquet(file_path)
        print(file_path)
        pic = np.zeros((256, 256, 1))
        x_array = np.sort(df["rho_c"].unique())
        y_array = np.sort(df["J"].unique())

        #smallest, largest = find_distances(x_array)
        #print(f"Smallest distance: {smallest}")
        #print(f"Largest distance: {largest}")
        #if not np.isclose(smallest, largest):
        #    print("Not close")

        for y_id, y_value in zip(range(255, -1, -1), reversed(y_array)):
            for x_id, x_value in zip(range(255, -1, -1), reversed(x_array)):
                df_reduced = df[np.isclose(df.rho_c, x_value) & np.isclose(df.J, y_value)]
                if len(df_reduced) == 0:
                    continue
                elif len(df_reduced) != 1:
                    continue
                value = df_reduced.r_ratio.values
                if np.isnan(value[0]):
                    continue
                elif value[0] > 1.00001:
                    continue
                pic[x_id, y_id, 0] = value[0]

        # Set J = 0 values to 1
        inside = False
        for id, value in enumerate(pic[:, 1, 0]):
            if inside:        
                pic[id, 0] = 1
            elif value != 0:
                inside = True
        data_2d = pic.reshape(256, 256)
        
        right_mask = np.zeros(data_2d.shape, dtype=bool)
        for row_id, row in enumerate(data_2d):
            for id in range(255, -1, -1):
                if row[id] != 0:
                    right_mask[row_id, id + 1:] = True
                    break

        y, x = np.mgrid[0:data_2d.shape[0], 0:data_2d.shape[1]]
        mask = data_2d != 0
        x1 = x[mask]
        y1 = y[mask]
        newarr = data_2d[mask]
        zero_mask = data_2d == 0
        
        interpolated = griddata((x1, y1), newarr, (x, y), method='cubic')
        interpolated = np.nan_to_num(interpolated, nan=0)

        filled = data_2d.copy()
        filled[zero_mask] = interpolated[zero_mask]
        filled[right_mask] = 0

        pic = filled.reshape(256, 256, 1)
        return pic
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    start_time = time.time()
    # Process 10 files

    directory_path = "/mnt/rafast/miler/new_normalized_star/test/"
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.parquet')]
    
    # Start Dask client
    client = Client(n_workers=50, threads_per_worker=1, memory_limit='4GB')  # For local cluster, replace with Client('scheduler-address') for remote clusters

    # Create a Dask Bag of files to process
    file_bag = db.from_sequence(files, npartitions=50)

    
    # Create a list of delayed tasks
    #tasks = [delayed(process_file)(file) for file in files]
    
    # Compute the results
    #results = dask.compute(*tasks)

    # Map the processing function to the files
    results_bag = file_bag.map(process_file)

    # Compute the results (this will execute in parallel)
    results = results_bag.compute()
    
    # Filter out None results
    pics = [result for result in results if result is not None]
    
    # Convert list of pics to numpy array
    pics = np.array(pics)
    
    # Save to a compressed file
    np.save('/mnt/rafast/miler/ml_data_pics.npy', pics)
    print(f"Saved pics to ml_data_pics.npy with shape {pics.shape}")
    end_time = time.time()
    time_per_file = (end_time - start_time) / len(files)
    print(time_per_file)

def find_distances(x_array):
    if len(x_array) < 2:
        raise ValueError("The list must contain at least two elements.")

    # Sort the list to ensure distances are calculated between adjacent elements
    sorted_array = sorted(x_array)
    
    # Initialize distances list
    distances = []

    # Calculate distances between neighboring elements
    for i in range(len(sorted_array) - 1):
        distance = abs(sorted_array[i + 1] - sorted_array[i])
        distances.append(distance)
    
    # Find the smallest and largest distance
    smallest_distance = min(distances)
    largest_distance = max(distances)
    
    return smallest_distance, largest_distance


if __name__ == "__main__":
    main()
