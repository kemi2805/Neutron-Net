# autoencoder.py
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import Layer, Conv2D, GroupNormalization, Attention, Activation, experimental
from keras import Input
from keras.models import Model
from keras.backend import set_image_data_format

from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float
    
def swish(x: tf.Tensor) -> tf.Tensor:
    return x * tf.sigmoid(x)

# Set the image data format globally
set_image_data_format('channels_last')

class AttnBlock(Layer):
    def __init__(self, out_channels: int):
        """
        Initialize the Attention Block.

        Args:
            out_channels (int): Number of output and input channels.
        """
        super(AttnBlock, self).__init__()
        self.out_channels = out_channels

        self.norm = experimental.preprocessing.Rescaling(scale=1./255)  # I do not really need it. This is a relic from real image

        # Convolutional layers for query, key, value, and output projection
        self.q = Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.k = Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.v = Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.proj_out = Conv2D(out_channels, kernel_size=1, use_bias=False)

    def attention(self, h_):
        """
        Apply the attention mechanism.

        Args:
            h_ (tf.Tensor): Input tensor of shape (batch_size, height, width, channels).

        Returns:
            tf.Tensor: Output tensor after applying attention mechanism.
        """
        h_ = self.norm(h_)  # Normalize input tensor
        q = self.q(h_)      # Compute query tensor
        k = self.k(h_)      # Compute key tensor
        v = self.v(h_)      # Compute value tensor

        _, h, w, c = q.shape
        q = tf.reshape(q, [-1, 1, h * w, c])  # Reshape query tensor
        k = tf.reshape(k, [-1, 1, h * w, c])  # Reshape key tensor
        v = tf.reshape(v, [-1, 1, h * w, c])  # Reshape value tensor

         # Scaled Dot Product Attention
        attn_logits = tf.matmul(q, k, transpose_b=True)  # Compute attention logits
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)  # Compute attention weights
        h_ = tf.matmul(attn_weights, v)  # Apply attention weights to value tensor

        return tf.reshape(h_, [-1, h, w, c])

    def call(self, x):
        """
        Forward pass through the Attention Block.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, height, width, channels).

        Returns:
            tf.Tensor: Output tensor after applying attention and projection.
        """
        return x + self.proj_out(self.attention(x))

# Define Swish activation function
class Swish(Layer):
    def __init__(self, beta=1.0, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        return inputs * tf.sigmoid(self.beta * inputs)

# Define the ResnetBlock class
class ResnetBlock(Layer):
    def __init__(self, in_channels: int, out_channels: int = None):
        """
        Initialize the ResNet Block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int, optional): Number of output channels. If None, it defaults to `in_channels`.
        """

        super(ResnetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = GroupNormalization(scale=1./255, groups=2)  # Placeholder for GroupNorm
        self.conv1 = Conv2D(self.out_channels, kernel_size=3, strides=1, padding='same', use_bias=False)
        
        self.norm2 = GroupNormalization(scale=1./255, groups=2)  # Placeholder for GroupNorm
        self.conv2 = Conv2D(self.out_channels, kernel_size=3, strides=1, padding='same', use_bias=False)
        
        self.swish = Swish()
        
        # Shortcut for adjusting dimensions when input and output channels differ
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv2D(self.out_channels, kernel_size=1, strides=1, padding='valid', use_bias=False)

    def call(self, x):
        """
        Forward pass through the ResNet Block.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, height, width, channels).

        Returns:
            tf.Tensor: Output tensor after applying ResNet block operations.
        """

        h = x
        h = self.norm1(h)
        h = self.swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h # Add residual connection and return the result
    
class Downsample(Layer):
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.conv = Conv2D(in_channels, kernel_size=3, strides=2, padding='valid', use_bias=False)

    def call(self, x):
        # Manual asymmetric padding
        pad = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])  # Last dimension is channel dimension, no padding
        x_padded = tf.pad(x, pad, mode='CONSTANT')
        x = self.conv(x_padded)
        return x

class Upsample(Layer):
    def __init__(self, in_channels):
        super(Upsample, self).__init__()
        self.conv = Conv2D(in_channels, kernel_size=3, strides=1, padding='same', use_bias=False)

    def call(self, x):
        # Upsampling using nearest neighbor interpolation
        x_upsampled = tf.image.resize(x, [x.shape[1]*2, x.shape[2]*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = self.conv(x_upsampled)
        return x
    
class Encoder(Model):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        """
        Initialize the Encoder network.

        Args:
            resolution (int): The spatial resolution of the input images.
            in_channels (int): Number of input channels.
            ch (int): Base number of channels for the initial convolution.
            ch_mult (list[int]): List of channel multipliers for each resolution level.
            num_res_blocks (int): Number of residual blocks per resolution level.
            z_channels (int): Number of channels in the output.
        """
        super(Encoder, self).__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ch_mult = ch_mult
        self.z_channels = z_channels
        
        # Downsampling: Initial convolution
        self.conv_in = Conv2D(ch, kernel_size=3, strides=1, padding='same')
        
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = []
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = []
            attn = []
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = {}
            down['block'] = block
            down['attn'] = attn
            if i_level != self.num_resolutions - 1:
                down['downsample'] = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle blocks: Residual and Attention blocks
        self.mid = {}
        self.mid['block_1'] = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid['attn_1'] = AttnBlock(block_in)
        self.mid['block_2'] = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # End: Output normalization and convolution
        self.norm_out = GroupNormalization(groups=2, axis=-1)
        self.conv_out = Conv2D(2 * z_channels, kernel_size=3, strides=1, padding='same')

    def call(self, x):
        """
        Forward pass through the Encoder network.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, height, width, channels).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, height, width, 2 * z_channels).
        """
        # Downsampling path
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level]['block'][i_block](h)
                if len(self.down[i_level]['attn']) > 0:
                    h = self.down[i_level]['attn'][i_block](h)
            if i_level != self.num_resolutions - 1:
                h= self.down[i_level]['downsample'](h)

        # Middle blocks
        h = self.mid['block_1'](h)
        h = self.mid['attn_1'](h)
        h = self.mid['block_2'](h)

        # End: Output normalization and activation
        h = self.norm_out(h)
        h = Activation('swish')(h)
        h = self.conv_out(h)
        return h

    def get_config(self):
        config = super().get_config()
        config.update({
            "resolution": self.resolution,
            "in_channels": self.in_channels,
            "ch": self.ch,
            "ch_mult": self.ch_mult,
            "num_res_blocks": self.num_res_blocks,
            "z_channels": self.z_channels
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Decoder(Model):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        """
        Initialize the Decoder network.

        Args:
            ch (int): Base number of channels for the initial convolution.
            out_ch (int): Number of output channels.
            ch_mult (list[int]): List of channel multipliers for each resolution level.
            num_res_blocks (int): Number of residual blocks per resolution level.
            in_channels (int): Number of input channels (typically the same as z_channels).
            resolution (int): The spatial resolution of the input tensor.
            z_channels (int): Number of channels in the input tensor.
        """
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ch_mult = ch_mult
        self.z_channels = z_channels
        self.out_ch = out_ch
        self.ffactor = 2 ** (self.num_resolutions - 1) # Factor for final upsampling

        # Compute the input channels and resolution at the lowest resolution level
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, curr_res, curr_res, z_channels)
        
        # Initial convolution: z to block_i
        self.conv_in = Conv2D(block_in, kernel_size=3, strides=1, padding='same')
        
        # Middle blocks: Residual and Attention blocks
        self.mid = {}
        self.mid['block_1'] = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid['attn_1'] = AttnBlock(block_in)
        self.mid['block_2'] = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # Upsampling path
        self.up  = []
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            attn = []
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = {}
            up['block'] = block
            up['attn'] = attn
            if i_level != self.num_resolutions - 1:
                up['upsample'] = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.append(up)

        # End: Output normalization and convolution
        self.norm_out = GroupNormalization(groups=2, axis=-1)
        self.conv_out = Conv2D(out_ch, kernel_size=3, strides=1, padding='same')
        
    def call(self, z):
        """
        Forward pass through the Decoder network.

        Args:
            z (tf.Tensor): Input tensor of shape (batch_size, height, width, z_channels).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, height, width, out_ch).
        """
        # Initial convolution: z to block_in
        h = self.conv_in(z)

        # Middle blocks
        h = self.mid['block_1'](h)
        h = self.mid['attn_1'](h)
        h = self.mid['block_2'](h)

        # Upsampling path
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level]['block'][i_block](h)
                if len(self.up[i_level]['attn']) > 0:
                    h = self.up[i_level]['attn'][i_block](h)
            if i_level != 0:
                h = self.up[i_level]['upsample'](h)

        # End: Output normalization and activation
        h = self.norm_out(h)
        h = Activation('swish')(h)
        h = self.conv_out(h)
        return h

    def get_config(self):
        config = super().get_config()
        config.update({
            "ch": self.ch,
            "out_ch": self.out_ch,
            "ch_mult": self.ch_mult,
            "num_res_blocks": self.num_res_blocks,
            "in_channels": self.in_channels,
            "resolution": self.resolution,
            "z_channels": self.z_channels
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
class DiagonalGaussian(Layer):
    def __init__(self, sample=True, chunk_dim=-1):
        super(DiagonalGaussian, self).__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def call(self, z):
        # Splitting the tensor into mean and logvar
        mean, logvar = tf.split(z, num_or_size_splits=2, axis=self.chunk_dim)

        # Sampling from the Gaussian distribution
        if self.sample:
            std = tf.exp(0.5 * logvar)
            return mean + std * tf.random.normal(shape=tf.shape(mean))
        else:
            return mean


class AutoEncoder(Model):
    def __init__(self, params):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x):
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z):
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def call(self, x):
        return self.decode(self.encode(x))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "scale_factor": self.scale_factor,
            "shift_factor": self.shift_factor,
            "encoder_params": {
                "resolution": self.encoder.resolution,
                "in_channels": self.encoder.in_channels,
                "ch": self.encoder.ch,
                "ch_mult": self.encoder.ch_mult,
                "num_res_blocks": self.encoder.num_res_blocks,
                "z_channels": self.encoder.z_channels
            },
            "decoder_params": {
                "resolution": self.decoder.resolution,
                "in_channels": self.decoder.in_channels,
                "ch": self.decoder.ch,
                "out_ch": self.decoder.out_ch,
                "ch_mult": self.decoder.ch_mult,
                "num_res_blocks": self.decoder.num_res_blocks,
                "z_channels": self.decoder.z_channels
            }
        })
        return config


def main():
    def train_autoencoder(autoencoder, train_data, val_data, epochs=20, batch_size=2):
        # Initialize optimizer, loss function, and metric function
        optimizer = Adam(learning_rate=0.001)
        loss_fn = MeanSquaredError()  # MSE loss function
        metric_fn = MeanAbsoluteError()  # MAE metric function

        # Compile the autoencoder
        autoencoder.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric_fn])

        # Build the model (input shape needs to match your data)
        autoencoder.build(input_shape=(1, 256, 256, 3))  # Adjust the input shape as needed

        # Define checkpoint path and other callbacks
        checkpoint_path = "/mnt/rafast/miler/checkpoint.tf"
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, verbose=1)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_delta=0.001)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Combine all callbacks
        callbacks = [checkpoint_callback, reduce_lr, early_stop]

        # Train the autoencoder
        history = autoencoder.fit(
            x=train_data,
            y=train_data,
            epochs=epochs,
            validation_data=(val_data, val_data),
            callbacks=callbacks,
            batch_size=batch_size,
            verbose=1
        )
        return history

    def plot_training_curves(history) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Autoencoder Training Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # Optionally, save in different formats
        plt.savefig("/mnt/rafast/miler/learning.pdf")  # Save as PDF file
        return
    
    params = AutoEncoderParams(
        resolution=256,
        in_channels=3,
        ch=32,
        out_ch=3,
        ch_mult=[1,2,4],
        num_res_blocks=3,
        z_channels=128,
        scale_factor=1.0,
        shift_factor=0.0
        )
    # Beispiel für die Verwendung des Codes
    autoencoder = AutoEncoder(params=params)

    def load_data():
        # Generieren von Training-Daten
        train_x = np.random.rand(100, 256, 256, 3).astype(np.float32)

        # Generieren von Validierungsdaten
        val_x = np.random.rand(20, 256, 256, 3).astype(np.float32)

        # Rückgabe als Dictionary mit Training-, Validierungs- und Testdaten
        return train_x, val_x

    train_data, val_data = load_data()

    history = train_autoencoder(autoencoder, train_data, val_data)

    plot_training_curves(history)

    def plot_original_vs_reconstructed(autoencoder, data, num_samples=5):
        # Predict reconstructed images
        try:
            reconstructed = autoencoder.predict(data[:num_samples])
        except Exception as e:
            print(f"Error during prediction: {e}")
            return

        # Create a figure with subplots
        fig, axes = plt.subplots(nrows=num_samples, ncols=2, figsize=(15, 3 * num_samples))

        # Normalize pixel values
        vmin = np.min([np.min(data), np.min(reconstructed)])
        vmax = np.max([np.max(data), np.max(reconstructed)])

        # Plot original and reconstructed images
        for i in range(num_samples):
            axes[i, 0].imshow(data[i], vmin=vmin, vmax=vmax)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(reconstructed[i], vmin=vmin, vmax=vmax)
            axes[i, 1].set_title('Reconstructed')
            axes[i, 1].axis('off')

        # Layout so plots do not overlap
        plt.tight_layout()
        plt.savefig("/mnt/rafast/miler/bad_plots.pdf")  # Save as PDF file
    # Usage
    plot_original_vs_reconstructed(autoencoder, train_data)
    return 0

if __name__ == "__main__":
    main()