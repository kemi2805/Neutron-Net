# main.py
from autoencoder import AutoEncoderParams, AutoEncoder
from training import train_autoencoder, plot_training_curves
from utils import load_random_data, plot_original_vs_reconstructed

def main():
    params = AutoEncoderParams(
        resolution=256,
        in_channels=3,
        ch=32,
        out_ch=3,
        ch_mult=[1,2,4,8],
        num_res_blocks=3,
        z_channels=256,
        scale_factor=1.0,
        shift_factor=0.0
    )
    
    autoencoder = AutoEncoder(params=params)
    
    train_data, val_data = load_random_data((256,256,3),100,20)
    
    history = train_autoencoder(autoencoder, train_data, val_data, epochs=50)
    
    plot_training_curves(history)
    
    plot_original_vs_reconstructed(autoencoder, train_data)

if __name__ == "__main__":
    main()
