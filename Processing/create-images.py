import numpy as np
import pandas as pd
import os
import dask
import dask.dataframe as dd
from dask.distributed import Client
import time

def process_file(file_path):
    try:
        df = pd.read_parquet(file_path)
        print(file_path)
        pic = np.zeros((256, 256, 1))
        x_array = np.sort(df["rho_c"].unique())
        y_array = np.sort(df["J"].unique())
        
        print(len(x_array))
        print(len(y_array))
        for y_id, y_value in enumerate(y_array):
            for x_id, x_value in enumerate(x_array):
                df_reduced = df[np.isclose(df.rho_c, x_value) & np.isclose(df.J, y_value)]
                if len(df_reduced) == 0:
                    continue
                if len(df_reduced) != 1:
                    continue
                value = df_reduced.r_ratio.values
                if np.isnan(value[0]):
                    continue
                pic[x_id, y_id, 0] = value[0]
        
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
    client = Client()  # For local cluster, replace with Client('scheduler-address') for remote clusters

    # Use dask.delayed to parallelize the file processing
    from dask import delayed
    
    # Create a list of delayed tasks
    tasks = [delayed(process_file)(file) for file in files]
    
    # Compute the results
    results = dask.compute(*tasks)
    
    # Filter out None results
    pics = [result for result in results if result is not None]
    
    # Convert list of pics to numpy array
    pics = np.array(pics)
    
    # Save to a compressed file
    np.savez_compressed('/mnt/rafast/miler/test.npz', pics=pics)
    print(f"Saved pics to test.npz with shape {pics.shape}")
    end_time = time.time()
    time_per_file = (end_time - start_time)
    print(time_per_file)

if __name__ == "__main__":
    main()
