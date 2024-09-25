# main.py
import tensorflow as tf
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
    max_batch_size = 2
    accumulation_steps = batch_size // max_batch_size

    beta_scheduler = CosineBetaScheduler(1000).cosine_schedule
     
    autoencoder = AutoEncoder(params=params)
    neutron_net = NeutronNet(params, beta_scheduler, 4)

    optimizer = Adam(learning_rate=0.001)
    loss_fn = MeanSquaredError()
    metric_fn = MeanAbsoluteError()
    
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

    len_data = data.shape[0]
    val_split = 0.1

    data_tensor = tf.convert_to_tensor(data)
    dataset = tf.data.Dataset.from_tensor_slices(data_tensor)
    # Calculate the number of training and validation samples
    val_size = int(len_data * val_split)
    train_size = len_data - val_size

    # Create the training and validation datasets
    train_dataset = dataset.take(train_size).batch(batch_size).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = dataset.skip(train_size).take(val_size).batch(batch_size).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    #dataset = dataset.interleave(lambda x: tf.data.Dataset.from_tensor_slices(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    neutron_net.train_model(train_dataset, epochs, callbacks, optimizer, val_dataset)

if __name__ == "__main__":
    main()