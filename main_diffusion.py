# main.py
from tensorflow.data import Dataset, experimental # type: ignore
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, History

import numpy as np

from autoencoder import AutoEncoderParams, AutoEncoder
from Diffusion.NeutronNet import NeutronNet, CosineBetaScheduler



def main():
    params = AutoEncoderParams(
        resolution=256,
        in_channels=1,
        ch=1,
        out_ch=1,
        ch_mult=[2,4,8,16],
        num_res_blocks=4,
        z_channels=4,
        scale_factor=1.0,
        shift_factor=0.0
    )
    epochs=100
    batch_size=8
    beta_scheduler = CosineBetaScheduler(1000).cosine_schedule
     
    autoencoder = AutoEncoder(params=params)
    neutron_net = NeutronNet(params, beta_scheduler, 4)
    
    patience = 10
    checkpoint_path = "/mnt/rafast/miler/some_model"
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience, min_delta=0.001)
    callbacks = [checkpoint_callback, reduce_lr, early_stopping]


    try:
        data = np.load("/mnt/rafast/miler/ml_data_pics.npy", mmap_mode='r')
        print("File loaded successfully!")


    except Exception as e:
        print(f"Error loading the file: {e}")

    dataset = Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(10000).batch(batch_size).cache().prefetch(buffer_size=experimental.AUTOTUNE)


    data = data.reshape(data.shape[0],256,256,1)
    train_data, val_data = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]

    # Initialize optimizer, loss function, and metric function
    optimizer = Adam(learning_rate=0.001)
    loss_fn = MeanSquaredError()
    metric_fn = MeanAbsoluteError()

    # Compile the autoencoder
    autoencoder.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric_fn])

    # Build the model with the specified input shape
    autoencoder.build(input_shape=(batch_size, 256, 256, 1))  # Adjust as needed

    print(autoencoder.summary())

    
    #plot_training_curves(history[-1])
    
    #plot_original_vs_reconstructed_griddata(autoencoder, val_data)

    print(autoencoder.encoder.summary())
    print(autoencoder.decoder.summary())
if __name__ == "__main__":
    main()