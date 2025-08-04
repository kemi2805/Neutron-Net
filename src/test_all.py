# test_all.py
import torch
import numpy as np
from src.models.autoencoder import AutoEncoder, AutoEncoderConfig
from src.models.diffusion import NeutronNet, CosineBetaScheduler
from src.data.data_loader import load_neutron_star_data
from src.utils.physics import validate_physics_constraints

def test_autoencoder():
    print("Testing AutoEncoder...")
    config = AutoEncoderConfig(resolution=64, ch_mult=[2, 4])
    model = AutoEncoder(config)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == x.shape
    print("✅ AutoEncoder test passed")

def test_diffusion():
    print("Testing Diffusion Model...")
    ae_config = AutoEncoderConfig(resolution=64, ch_mult=[2, 4])
    scheduler = CosineBetaScheduler(100)
    beta_schedule = scheduler.get_schedule()
    model = NeutronNet(ae_config, beta_schedule)
    
    x = torch.randn(2, 1, 64, 64)
    t = torch.randint(0, 100, (2,))
    noisy, noise = model.forward_diffusion_step(x, t)
    pred = model.reverse_diffusion_step(noisy, t)
    assert pred.shape == x.shape
    print("✅ Diffusion model test passed")

def test_data_loading():
    print("Testing Data Loading...")
    # Test with dummy data
    dummy_data = np.random.rand(10, 64, 64).astype(np.float32)
    np.save('test_data.npy', dummy_data)
    
    train_loader, val_loader = load_neutron_star_data(
        'test_data.npy',
        batch_size=2,
        validation_split=0.2
    )
    
    batch = next(iter(train_loader))
    assert batch.shape == (2, 1, 64, 64)
    print("✅ Data loading test passed")
    
    import os
    os.remove('test_data.npy')

if __name__ == "__main__":
    test_autoencoder()
    test_diffusion()
    test_data_loading()
    print("\n✅ All tests passed!")