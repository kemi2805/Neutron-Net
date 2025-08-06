#!/usr/bin/env python3
"""
Standalone test script for GPU, simple Conv Autoencoder, forward pass, and data loading.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

def test_gpu_setup():
    print("🔍 Testing GPU Setup...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")

        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = torch.matmul(x, y)
        print(f"✅ Basic GPU operations work: {z.shape} on {z.device}")

        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated()
        print(f"GPU Memory - Total: {total_memory/1e9:.1f}GB, Allocated: {allocated/1e6:.1f}MB")

        return True
    else:
        print("❌ CUDA not available!")
        return False

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 1 -> 8 channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),# 16x16 -> 8x8
            nn.ReLU(),
        )
        # Decoder: 32 -> 1 channel
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),   # 32x32 -> 64x64
            nn.Sigmoid(),  # output normalized to [0,1]
        )

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.encoder(x)
        print(f"After encoder: {x.shape}")
        x = self.decoder(x)
        print(f"After decoder: {x.shape}")
        return x

def test_model_creation(device):
    print("\n🏗️ Testing Model Creation...")
    try:
        model = SimpleAutoencoder()
        model = model.to(device)
        print(f"✅ Model created successfully on {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_forward_pass(model, device):
    print("\n⚡ Testing Forward Pass...")
    if model is None:
        print("❌ No model to test")
        return False
    try:
        batch_size = 2
        x = torch.randn(batch_size, 1, 64, 64, device=device)
        model.eval()
        with torch.no_grad():
            output = model(x)
        print(f"✅ Forward pass successful: {x.shape} -> {output.shape}")
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    print("\n📁 Testing Data Loading...")

    data_path = "/mnt/rafast/miler/ml_data_pics.npy"
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        return False
    try:
        data = np.load(data_path, mmap_mode='r')
        print(f"✅ Data loaded: {data.shape}, dtype: {data.dtype}")
        print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")

        sample = torch.from_numpy(data[:4].astype(np.float32))
        if len(sample.shape) == 3:
            sample = sample.unsqueeze(1)  # add channel dim
        elif len(sample.shape) == 4 and sample.shape[-1] == 1:
            sample = sample.permute(0, 3, 1, 2)  # NHWC -> NCHW

        print(f"✅ Torch conversion: {sample.shape}")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def main():
    print("🧪 Running GPU and Model Tests...")
    print("=" * 50)

    gpu_ok = test_gpu_setup()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = test_model_creation(device)
    import os
    print("MIOPEN_USER_DB_PATH =", os.environ.get("MIOPEN_USER_DB_PATH"))
    forward_ok = test_forward_pass(model, device)
    data_ok = test_data_loading()

    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"GPU Setup: {'✅' if gpu_ok else '❌'}")
    print(f"Model Creation: {'✅' if model is not None else '❌'}")
    print(f"Forward Pass: {'✅' if forward_ok else '❌'}")
    print(f"Data Loading: {'✅' if data_ok else '❌'}")

    if all([gpu_ok, model is not None, forward_ok, data_ok]):
        print("\n🎉 All tests passed! Ready for training.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the issues above before training.")
        return False

if __name__ == "__main__":
    main()
