# main.py
from autoencoder import AutoEncoderParams, AutoEncoder
from training import train_autoencoder, plot_training_curves
from utils import load_random_data, plot_original_vs_reconstructed, process_all_eos, plot_grid_data, plot_original_vs_reconstructed_griddata, plot_encoder_filters
import numpy as np

def main():
    params = AutoEncoderParams(
        resolution=200,
        in_channels=3,
        ch=32,
        out_ch=1,
        ch_mult=[1,2,4],
        num_res_blocks=3,
        z_channels=128,
        scale_factor=1.0,
        shift_factor=0.0
    )
    
    autoencoder = AutoEncoder(params=params)
    
    train_data, val_data = load_random_data((256,256,3),100,20)
    data = np.load("/mnt/rafast/miler/grid_array.npy")
    data = data.reshape(data.shape[0],200,200,1)
    
    train_data, val_data = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]

    history = train_autoencoder(autoencoder, train_data, val_data, epochs=5, batch_size=4)
    
    plot_training_curves(history)
    
    plot_original_vs_reconstructed_griddata(autoencoder, val_data)

    print(autoencoder.encoder.summary())
    print(autoencoder.decoder.summary())

def test_data():
    train_data, val_data = process_all_eos('/mnt/rafast/miler/Some_Star.parquet')
    plot_grid_data(train_data)

def plot_data():
    params = AutoEncoderParams(
        resolution=200,
        in_channels=3,
        ch=32,
        out_ch=1,
        ch_mult=[1,2,4],
        num_res_blocks=3,
        z_channels=128,
        scale_factor=1.0,
        shift_factor=0.0
    )
    autoencoder = AutoEncoder(params=params)

    checkpoint_path = "/mnt/rafast/miler/checkpoint.tf"
    train_data, val_data = load_random_data((256,256,3),100,20)
    data = np.load("/mnt/rafast/miler/grid_array.npy")
    data = data.reshape(data.shape[0],200,200,1)
    
    train_data, val_data = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]

    # Step 3: Load the weights from the checkpoint
    autoencoder.load_weights(checkpoint_path)

    plot_encoder_filters(autoencoder.encoder, val_data, num_samples=1)
if __name__ == "__main__":
    plot_data()
