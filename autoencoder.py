# autoencoder.py
from keras.models import Model

from dataclasses import dataclass
from layers import Encoder, Decoder, DiagonalGaussian

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