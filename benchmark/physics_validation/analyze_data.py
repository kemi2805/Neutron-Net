# benchmark/physics_validation/analyze_data.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from src.utils.visualization import plot_grid_data, compare_distributions
from src.utils.physics import validate_physics_constraints

import seaborn as sns

def analyze_neutron_star_data(data_path, output_dir='analysis_results'):
    """Comprehensive analysis of neutron star dataset."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}")
    data = np.load(data_path)
    print(f"Data shape: {data.shape}")
    print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")
    
    # Basic statistics
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Number of samples: {data.shape[0]}")
    print(f"Image dimensions: {data.shape[1]} x {data.shape[2]}")
    print(f"Mean value: {data.mean():.4f}")
    print(f"Std deviation: {data.std():.4f}")
    print(f"Median value: {np.median(data):.4f}")
    
    # Check for physical validity
    valid_range = (data >= 0) & (data <= 1)
    valid_fraction = valid_range.mean()
    print(f"Fraction in valid [0,1] range: {valid_fraction:.2%}")
    
    # Non-zero analysis
    nonzero_mask = data > 0
    nonzero_fraction = nonzero_mask.mean()
    print(f"Non-zero fraction: {nonzero_fraction:.2%}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Sample images
    for i in range(3):
        if i < data.shape[0]:
            im = axes[0, i].imshow(data[i], cmap='viridis', aspect='auto')
            axes[0, i].set_title(f'Sample {i+1}')
            axes[0, i].axis('off')
            plt.colorbar(im, ax=axes[0, i])
    
    # 2. Mean image
    mean_image = data.mean(axis=0)
    im = axes[1, 0].imshow(mean_image, cmap='viridis', aspect='auto')
    axes[1, 0].set_title('Mean Image')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 3. Std image
    std_image = data.std(axis=0)
    im = axes[1, 1].imshow(std_image, cmap='viridis', aspect='auto')
    axes[1, 1].set_title('Std Deviation')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])
    
    # 4. Value distribution
    axes[1, 2].hist(data.flatten(), bins=100, density=True, alpha=0.7)
    axes[1, 2].set_xlabel('r_ratio value')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Value Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Radial analysis
    print("\n" + "="*50)
    print("RADIAL STRUCTURE ANALYSIS")
    print("="*50)
    
    # Compute average radial profile
    center_y, center_x = data.shape[1] // 2, data.shape[2] // 2
    y, x = np.ogrid[:data.shape[1], :data.shape[2]]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Bin by radius
    max_radius = min(center_x, center_y)
    radial_bins = np.linspace(0, max_radius, 50)
    radial_profile = []
    
    for i in range(len(radial_bins) - 1):
        mask = (r >= radial_bins[i]) & (r < radial_bins[i+1])
        if mask.any():
            radial_profile.append(data[:, mask].mean())
        else:
            radial_profile.append(0)
    
    # Plot radial profile
    plt.figure(figsize=(10, 6))
    plt.plot(radial_bins[:-1], radial_profile, 'b-', linewidth=2)
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Mean r_ratio')
    plt.title('Average Radial Profile')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'radial_profile.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation analysis
    print("\nComputing spatial correlations...")
    
    # Sample a subset for correlation analysis
    sample_size = min(100, data.shape[0])
    sample_indices = np.random.choice(data.shape[0], sample_size, replace=False)
    
    # Flatten spatial dimensions
    flattened = data[sample_indices].reshape(sample_size, -1)
    
    # Compute correlation matrix for a subset of pixels
    pixel_subset = np.random.choice(flattened.shape[1], 100, replace=False)
    corr_matrix = np.corrcoef(flattened[:, pixel_subset].T)
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Pixel Correlation Matrix (subset)')
    plt.savefig(output_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary report
    with open(output_path / 'analysis_report.txt', 'w') as f:
        f.write("NEUTRON STAR DATA ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Shape: {data.shape}\n")
        f.write(f"Total samples: {data.shape[0]}\n")
        f.write(f"Image size: {data.shape[1]}x{data.shape[2]}\n\n")
        
        f.write("Statistics:\n")
        f.write(f"  Mean: {data.mean():.6f}\n")
        f.write(f"  Std: {data.std():.6f}\n")
        f.write(f"  Min: {data.min():.6f}\n")
        f.write(f"  Max: {data.max():.6f}\n")
        f.write(f"  Median: {np.median(data):.6f}\n\n")
        
        f.write("Physical Validity:\n")
        f.write(f"  Values in [0,1]: {valid_fraction:.2%}\n")
        f.write(f"  Non-zero values: {nonzero_fraction:.2%}\n")
        
        # Compute percentiles
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        f.write("\nPercentiles:\n")
        for p in percentiles:
            value = np.percentile(data, p)
            f.write(f"  {p}%: {value:.6f}\n")
    
    print(f"\nAnalysis complete! Results saved to {output_path}")
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze neutron star data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data file')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='Output directory')
    
    args = parser.parse_args()
    analyze_neutron_star_data(args.data_path, args.output_dir)