#!/usr/bin/env python3
"""
Test script to verify GPU setup and model initialization before full training.
"""

import os
import torch
import numpy as np
from pathlib import Path

def test_gpu_setup():
    """Test basic GPU functionality."""
    print("🔍 Testing GPU Setup...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Test basic operations
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = torch.matmul(x, y)
        print(f"✅ Basic GPU operations work: {z.shape} on {z.device}")
        
        # Test memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated()
        print(f"GPU Memory - Total: {total_memory/1e9:.1f}GB, Allocated: {allocated/1e6:.1f}MB")
        
        return True
    else:
        print("❌ CUDA not available!")
        return False

def test_model_creation():
    """Test model creation without training."""
    print("\n🏗️ Testing Model Creation...")
    
    try:
        # Add project root to path
        import sys
        sys.path.append('/mnt/rafast/miler/codes/Neutron-Net')
        
        from src.models.autoencoder import AutoEncoder, AutoEncoderConfig
        from src.models.diffusion import NeutronNet, CosineBetaScheduler
        
        # Create small model for testing
        config = AutoEncoderConfig(
            resolution=64,  # Smaller for testing
            in_channels=1,
            ch=2,
            out_ch=1,
            ch_mult=[2, 4],  # Simpler architecture
            num_res_blocks=2,
            z_channels=4
        )
        
        # Create model
        scheduler = CosineBetaScheduler(100)  # Smaller for testing
        beta_schedule = scheduler.get_schedule()
        model = NeutronNet(config, beta_schedule, accumulation_steps=2)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"✅ Model created successfully on {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, device
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_forward_pass(model, device):
    """Test forward pass without training."""
    print("\n⚡ Testing Forward Pass...")
    
    if model is None:
        print("❌ No model to test")
        return False
    
    try:
        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 1, 64, 64, device=device)
        t = torch.randint(0, 100, (batch_size,), device=device)
        
        model.eval()
        with torch.no_grad():
            # Test autoencoder
            reconstructed = model.model(x)
            print(f"✅ Autoencoder forward pass: {x.shape} -> {reconstructed.shape}")
            
            # Test diffusion forward step
            noisy_images, noise = model.forward_diffusion_step(x, t)
            print(f"✅ Forward diffusion: {x.shape} -> {noisy_images.shape}")
            
            # Test reverse diffusion step
            predicted_noise = model.reverse_diffusion_step(noisy_images, t)
            print(f"✅ Reverse diffusion: {noisy_images.shape} -> {predicted_noise.shape}")
        
        print("✅ All forward passes successful!")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading."""
    print("\n📁 Testing Data Loading...")
    
    data_path = "/mnt/rafast/miler/ml_data_pics.npy"
    
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        return False
    
    try:
        # Load a small sample
        data = np.load(data_path, mmap_mode='r')
        print(f"✅ Data loaded: {data.shape}, dtype: {data.dtype}")
        print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")
        
        # Test conversion to torch
        sample = torch.from_numpy(data[:4].astype(np.float32))
        if len(sample.shape) == 3:
            sample = sample.unsqueeze(1)  # Add channel dim
        elif len(sample.shape) == 4 and sample.shape[-1] == 1:
            sample = sample.permute(0, 3, 1, 2)  # NHWC -> NCHW
            
        print(f"✅ Torch conversion: {sample.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running GPU and Model Tests...")
    print("=" * 50)
    
    # Test GPU
    gpu_ok = test_gpu_setup()
    
    # Test model creation
    model, device = test_model_creation()
    
    # Test forward pass
    forward_ok = test_forward_pass(model, device)
    
    # Test data loading
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
