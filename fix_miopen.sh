#!/bin/bash

# Fix MIOpen cache directory permissions and setup
echo "🔧 Fixing MIOpen cache directory..."

# Set environment variables
export MIOPEN_CACHE_DIR=/mnt/rafast/miler/miopen_cache
export MIOPEN_USER_DB_PATH=$MIOPEN_CACHE_DIR/miopen_user.db
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_CACHE_DIR

# Create cache directory with proper permissions
echo "📁 Creating MIOpen cache directory..."
mkdir -p $MIOPEN_CACHE_DIR
chmod -R 755 $MIOPEN_CACHE_DIR

# Remove any existing database files that might be corrupted
echo "🗑️ Cleaning existing cache files..."
rm -f $MIOPEN_CACHE_DIR/*.db*
rm -f $MIOPEN_CACHE_DIR/*.lock
rm -rf $MIOPEN_CACHE_DIR/kernels

# Create fresh cache directory structure
mkdir -p $MIOPEN_CACHE_DIR/kernels
chmod -R 755 $MIOPEN_CACHE_DIR

# Set additional ROCm environment variables
export ROCM_PATH=/opt/rocm-6.2.1
export HIP_VISIBLE_DEVICES=0
export MIOPEN_FIND_MODE=1
export MIOPEN_DEBUG_DISABLE_FIND_DB=0

echo "✅ MIOpen cache directory setup complete!"
echo "Cache directory: $MIOPEN_CACHE_DIR"
echo "Permissions: $(ls -ld $MIOPEN_CACHE_DIR)"

# Test GPU availability
echo "🔍 Testing GPU availability..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name()}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    # Test tensor operations
    x = torch.randn(2, 3).cuda()
    y = x + 1
    print(f'GPU tensor test: {y.shape} on {y.device}')
else:
    print('No CUDA device found!')
"
