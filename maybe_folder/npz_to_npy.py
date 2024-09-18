import numpy as np

data = np.load("/mnt/rafast/miler/ml_data_pics.npz", mmap_mode='r')
data = data[data.files[0]]
np.save("/mnt/rafast/miler/ml_data_pics.npy", data)
