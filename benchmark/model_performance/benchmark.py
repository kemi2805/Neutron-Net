# benchmark/benchmark.py
import torch
import time
import numpy as np
from src.models.autoencoder import AutoEncoder, AutoEncoderConfig
from src.models.diffusion import NeutronNet, CosineBetaScheduler
from src.utils.utilities import get_model_summary, print_model_summary

def benchmark_model(model, input_shape, num_iterations=100, device='cuda'):
    """Benchmark model inference and training speed."""
    model = model.to(device)
    model.eval()
    
    # Warmup
    dummy_input = torch.randn(input_shape).to(device)
    for _ in range(10):
        _ = model(dummy_input)
    
    # Inference benchmark
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    inference_time = (time.time() - start_time) / num_iterations
    
    # Training benchmark
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = output.mean()
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize() if device == 'cuda' else None
    training_time = (time.time() - start_time) / num_iterations
    
    return {
        'inference_time_ms': inference_time * 1000,
        'training_time_ms': training_time * 1000,
        'throughput_inf': input_shape[0] / inference_time,
        'throughput_train': input_shape[0] / training_time
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test different model configurations
    configs = [
        {'name': 'Small', 'resolution': 64, 'ch_mult': [2, 4], 'batch_size': 16},
        {'name': 'Medium', 'resolution': 128, 'ch_mult': [2, 4, 8], 'batch_size': 8},
        {'name': 'Large', 'resolution': 256, 'ch_mult': [2, 4, 8, 16], 'batch_size': 4},
    ]
    
    print("\n" + "="*60)
    print("AUTOENCODER BENCHMARKS")
    print("="*60)
    
    for cfg in configs:
        print(f"\n{cfg['name']} Model (resolution={cfg['resolution']}):")
        
        # Create model
        ae_config = AutoEncoderConfig(
            resolution=cfg['resolution'],
            ch_mult=cfg['ch_mult'],
            num_res_blocks=2,
            z_channels=4
        )
        model = AutoEncoder(ae_config)
        
        # Print model summary
        input_shape = (cfg['batch_size'], 1, cfg['resolution'], cfg['resolution'])
        print_model_summary(model, input_shape)
        
        # Benchmark
        results = benchmark_model(model, input_shape, num_iterations=50, device=str(device))
        
        print(f"\nPerformance Metrics:")
        print(f"  Inference time: {results['inference_time_ms']:.2f} ms/batch")
        print(f"  Training time: {results['training_time_ms']:.2f} ms/batch")
        print(f"  Inference throughput: {results['throughput_inf']:.1f} samples/sec")
        print(f"  Training throughput: {results['throughput_train']:.1f} samples/sec")
    
    print("\n" + "="*60)
    print("DIFFUSION MODEL BENCHMARKS")
    print("="*60)
    
    # Test diffusion model
    ae_config = AutoEncoderConfig(resolution=128, ch_mult=[2, 4, 8])
    scheduler = CosineBetaScheduler(1000)
    beta_schedule = scheduler.get_schedule()
    
    diffusion_model = NeutronNet(ae_config, beta_schedule)
    
    input_shape = (4, 1, 128, 128)
    print_model_summary(diffusion_model, input_shape)
    
    # Memory usage
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Forward pass
        dummy_input = torch.randn(input_shape).to(device)
        t = torch.randint(0, 1000, (input_shape[0],)).to(device)
        
        with torch.no_grad():
            _ = diffusion_model(dummy_input, t)
        
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak GPU memory usage: {memory_allocated:.2f} GB")

if __name__ == "__main__":
    main()