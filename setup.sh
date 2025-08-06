#!/bin/bash

# SSL for Vision Transformers (ViT) - Setup Script
# This script sets up the complete environment for the SSL ViT project

set -e  # Exit on any error

echo "🚀 Setting up SSL for Vision Transformers (ViT) project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the root directory of the SSL-for-VIT project"
    exit 1
fi

print_status "Setting up external dependencies..."

# Create external directory if it doesn't exist
mkdir -p external

# Clone iBOT repository
if [ ! -d "external/ibot" ]; then
    print_status "Cloning iBOT repository..."
    git clone https://github.com/bytedance/ibot.git external/ibot
    print_success "iBOT repository cloned successfully"
else
    print_warning "iBOT repository already exists, skipping..."
fi

# Clone DINO v2 repository
if [ ! -d "external/dino" ]; then
    print_status "Cloning DINO v2 repository..."
    git clone https://github.com/facebookresearch/dinov2.git external/dino
    print_success "DINO v2 repository cloned successfully"
else
    print_warning "DINO v2 repository already exists, skipping..."
fi

# Clone MAE repository (if needed)
if [ ! -d "external/mae" ]; then
    print_status "Cloning MAE repository..."
    git clone https://github.com/facebookresearch/mae.git external/mae
    print_success "MAE repository cloned successfully"
else
    print_warning "MAE repository already exists, skipping..."
fi

print_status "Setting up Python environment..."

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Install additional dependencies for external repositories
print_status "Installing additional dependencies for external repositories..."

# Install iBOT dependencies
if [ -f "external/ibot/requirements.txt" ]; then
    print_status "Installing iBOT dependencies..."
    pip install -r external/ibot/requirements.txt
fi

# Install DINO dependencies
if [ -f "external/dino/requirements.txt" ]; then
    print_status "Installing DINO dependencies..."
    pip install -r external/dino/requirements.txt
fi

# Install MAE dependencies
if [ -f "external/mae/requirements.txt" ]; then
    print_status "Installing MAE dependencies..."
    pip install -r external/mae/requirements.txt
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data/cifar10_splits
mkdir -p models/vit
mkdir -p models/ibot
mkdir -p models/dino
mkdir -p results/vit
mkdir -p results/ibot
mkdir -p results/dino
mkdir -p outputs/vit
mkdir -p outputs/ibot
mkdir -p outputs/dino

print_status "Setting up data directory structure..."
echo "# CIFAR-10 Data Directory" > data/README.md
echo "" >> data/README.md
echo "This directory should contain the CIFAR-10 dataset files:" >> data/README.md
echo "- cifar-10-python.tar.gz (original dataset)" >> data/README.md
echo "- cifar10_splits/ (processed splits)" >> data/README.md
echo "  - train_images.npy" >> data/README.md
echo "  - train_labels.npy" >> data/README.md
echo "  - val_images.npy" >> data/README.md
echo "  - val_labels.npy" >> data/README.md
echo "  - test_images.npy" >> data/README.md
echo "  - test_labels.npy" >> data/README.md
echo "  - probe_images.npy" >> data/README.md
echo "  - probe_labels.npy" >> data/README.md

print_status "Downloading CIFAR-10 dataset..."
if [ ! -f "data/cifar-10-python.tar.gz" ]; then
    wget -O data/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    print_success "CIFAR-10 dataset downloaded"
else
    print_warning "CIFAR-10 dataset already exists, skipping download..."
fi

# Extract CIFAR-10 if not already extracted
if [ ! -d "data/cifar-10-batches-py" ]; then
    print_status "Extracting CIFAR-10 dataset..."
    tar -xzf data/cifar-10-python.tar.gz -C data/
    print_success "CIFAR-10 dataset extracted"
else
    print_warning "CIFAR-10 dataset already extracted, skipping..."
fi

# Create data splits if they don't exist
if [ ! -f "data/cifar10_splits/train_images.npy" ]; then
    print_status "Creating CIFAR-10 data splits..."
    python scripts/data_download_and_split/create_cifar10_splits.py
    print_success "CIFAR-10 data splits created"
else
    print_warning "CIFAR-10 data splits already exist, skipping..."
fi

print_status "Setting up SLURM scripts..."
# Make SLURM scripts executable
find scripts -name "*.sh" -exec chmod +x {} \;

print_success "Setup completed successfully! 🎉"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run a quick test to verify setup:"
echo "   python scripts/vit/train_vit.py --mode supervised"
echo ""
echo "3. For SLURM jobs, use the scripts in the scripts/ directory"
echo ""
echo "📚 Documentation:"
echo "- README.md: Main project documentation"
echo "- docs/SETUP_INSTRUCTIONS.md: Detailed setup instructions"
echo "- docs/EXTERNAL_DEPENDENCIES.md: External repository information"
echo ""
echo "🔧 External repositories:"
echo "- iBOT: external/ibot/"
echo "- DINO v2: external/dino/dinov2/"
echo "- MAE: external/mae/"
echo ""
print_success "Happy training! 🚀" 