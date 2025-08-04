# Neutron-Net: Diffusion Models for Neutron Star Data

A PyTorch implementation of diffusion models specifically designed for neutron star r_ratio data, with physics-aware constraints and specialized architectures.

## Overview

This project implements a denoising diffusion model with an autoencoder backbone for generating and analyzing neutron star data. The model incorporates physics constraints specific to neutron star properties, ensuring generated data respects physical laws.

## Features

- **Physics-Aware Architecture**: Custom layers that enforce neutron star physics constraints
- **Autoencoder-based Diffusion**: Efficient latent space diffusion using a specialized autoencoder
- **Comprehensive Data Pipeline**: Augmentation strategies preserving physical properties
- **Multi-GPU Support**: Distributed training with ROCm/CUDA support
- **Extensive Visualization**: Tools for analyzing model outputs and physics validation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Neutron-Net.git
cd Neutron-Net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Support

For AMD GPUs (ROCm):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

For NVIDIA GPUs (CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Training a Model

```bash
python src/main.py \
    --data_path /path/to/ml_data_pics.npy \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 1e-3 \
    --checkpoint_dir checkpoints
```

### Using Pre-trained Models

```python
from src.models.autoencoder import AutoEncoder, AutoEncoderConfig
from src.models.diffusion import NeutronNet, CosineBetaScheduler
import torch

# Load model
config = AutoEncoderConfig(resolution=256)
model = NeutronNet(config, beta_schedule)
model.load_state_dict(torch.load('checkpoints/best_model.pt'))

# Generate samples
samples = model.sample(num_samples=10)
```

## Project Structure

```
Neutron-Net/
├── src/
│   ├── models/
│   │   ├── autoencoder.py    # VAE architecture
│   │   ├── diffusion.py      # Diffusion model
│   │   ├── layers.py         # Physics-aware layers
│   │   └── sampling.py       # Sampling algorithms
│   ├── data/
│   │   ├── data_loader.py    # Data loading utilities
│   │   ├── augmentation.py   # Physics-preserving augmentations
│   │   └── processing.py     # Data preprocessing
│   ├── training/
│   │   └── trainer.py        # Training loops and optimization
│   ├── utils/
│   │   ├── physics.py        # Physics validation metrics
│   │   ├── visualization.py  # Plotting and analysis tools
│   │   ├── logging.py        # Logging utilities
│   │   └── utilities.py      # General utilities
│   └── main.py               # Main training script
├── config/
│   ├── config.yaml           # Main configuration
│   ├── model/                # Model configurations
│   ├── training/             # Training configurations
│   └── data/                 # Data configurations
└── requirements.txt

```

## Configuration

The project uses Hydra/OmegaConf for configuration management. Modify `config/config.yaml` for global settings:

```yaml
project_name: "neutron_star_diffusion"
experiment_name: "baseline_v1"

data:
  root_path: "/path/to/your/data"
  processed_data: "${data.root_path}/ml_data_pics.npy"

training:
  epochs: 100
  batch_size: 8
  learning_rate: 1e-3

model:
  resolution: 256
  ch_mult: [2, 4, 8, 16]
  num_res_blocks: 4
```

## Physics Constraints

The model enforces several physics constraints specific to neutron stars:

1. **r_ratio bounds**: Values constrained to [0, 1]
2. **Energy conservation**: Total energy preserved across transformations
3. **Smoothness**: Gradients bounded to ensure physical plausibility
4. **Mass-radius consistency**: Ensures generated data follows neutron star mass-radius relations

## Testing

Run comprehensive tests:

```bash
# Test all components
python -m src.test_all

# Test individual modules
python -m src.models.autoencoder
python -m src.models.diffusion
python -m src.utils.physics
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{neutronnet2024,
  title={Neutron-Net: Physics-Aware Diffusion Models for Neutron Star Data},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

## Related Publications

This work builds upon the following neutron star research:
- Musolino, C., Ecker, C., & Rezzolla, L. (2023). "On the maximum mass and oblateness of rotating neutron stars with generic equations of state"

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Based on the denoising diffusion probabilistic models (DDPM) framework
- Incorporates physics constraints from neutron star literature
- Uses PyTorch and various scientific computing libraries