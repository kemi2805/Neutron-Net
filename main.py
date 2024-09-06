# main.py
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, History

from autoencoder import AutoEncoderParams, AutoEncoder
from training import train_autoencoder, plot_training_curves
from utils import process_all_eos
from utils_plot import plot_grid_data, plot_original_vs_reconstructed_griddata, plot_encoder_filters
import numpy as np

def main():
    params = AutoEncoderParams(
        resolution=200,
        in_channels=3,
        ch=32,
        out_ch=1,
        ch_mult=[1,2,4],
        num_res_blocks=5,
        z_channels=16,
        scale_factor=1.0,
        shift_factor=0.0
    )
    epochs=300
    batch_size=2
    
    autoencoder = AutoEncoder(params=params)
    
    data = np.load("/mnt/rafast/miler/grid_array.npy")
    data = data.reshape(data.shape[0],200,200,1)
    train_data, val_data = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]

    # Initialize optimizer, loss function, and metric function
    optimizer = Adam(learning_rate=0.001)
    loss_fn = MeanSquaredError()
    metric_fn = MeanAbsoluteError()

    # Compile the autoencoder
    autoencoder.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric_fn])

    # Build the model with the specified input shape
    autoencoder.build(input_shape=(batch_size, 200, 200, 1))  # Adjust as needed

    # Define callbacks
    checkpoint_path = "/mnt/rafast/miler/checkpoint_300.tf"
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_delta=0.001)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    callbacks = [checkpoint_callback, reduce_lr, early_stop]

    history = train_autoencoder(autoencoder, train_data, val_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    
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
        num_res_blocks=5,
        z_channels=16,
        scale_factor=1.0,
        shift_factor=0.0
    )
    autoencoder = AutoEncoder(params=params)

    checkpoint_path = "/mnt/rafast/miler/checkpoint_300.tf"
    train_data, val_data = load_random_data((256,256,3),100,20)
    data = np.load("/mnt/rafast/miler/grid_array.npy")
    data = data.reshape(data.shape[0],200,200,1)
    
    train_data, val_data = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]
    val_data = np.ones(shape=val_data.shape)

    # Step 3: Load the weights from the checkpoint
    print("------------------------------")
    autoencoder.load_weights(checkpoint_path)
    print("Wheigts were loaded")
    plot_encoder_filters(autoencoder.encoder, val_data, num_samples=2)
if __name__ == "__main__":
    plot_data()
