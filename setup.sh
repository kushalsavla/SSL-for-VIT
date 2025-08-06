#!/bin/bash

# SSL for Vision Transformers (ViT) - Complete Setup Script
# This script sets up the complete environment for the SSL ViT project

set -e  # Exit on any error

echo "🚀 Setting up SSL for Vision Transformers (ViT) project..."
echo "=================================================="

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

print_status "Creating project directory structure..."

# Create necessary directories
mkdir -p data/cifar10_splits
mkdir -p models/{vit,dino,mae,ibot}
mkdir -p results/{final_evaluation,qualitative_analysis}
mkdir -p outputs
mkdir -p external

print_success "Directory structure created"

print_status "Setting up external dependencies..."

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

# Clone MAE repository
if [ ! -d "external/mae" ]; then
    print_status "Cloning MAE repository..."
    git clone https://github.com/facebookresearch/mae.git external/mae
    print_success "MAE repository cloned successfully"
else
    print_warning "MAE repository already exists, skipping..."
fi

print_status "Setting up Python environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    print_status "Conda detected. Creating conda environment..."
    
    # Create conda environment if it doesn't exist
    if ! conda env list | grep -q "rl_project"; then
        conda create -n rl_project python=3.10 -y
        print_success "Conda environment 'rl_project' created"
    else
        print_warning "Conda environment 'rl_project' already exists"
    fi
    
    # Activate conda environment
    eval "$(conda shell.bash hook)"
    conda activate rl_project
    print_success "Activated conda environment: rl_project"
    
    # Upgrade pip and install requirements in conda environment
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    print_status "Installing Python dependencies in conda environment..."
    pip install -r requirements.txt
    
else
    print_status "Conda not found. Using virtual environment..."
    
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
    
    # Upgrade pip and install requirements in virtual environment
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    print_status "Installing Python dependencies in virtual environment..."
    pip install -r requirements.txt
fi

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

print_success "Setup completed successfully! 🎉"
echo ""
echo "=================================================="
echo "📋 Next Steps:"
echo "=================================================="
echo ""
echo "1. Activate the environment:"
if command -v conda &> /dev/null; then
    echo "   conda activate rl_project"
else
    echo "   source venv/bin/activate"
fi
echo ""
echo "2. Run experiments:"
echo "   # Test on unseen data"
echo "   sbatch scripts/analysis/test_unseen_data.sh"
echo ""
echo "   # Qualitative analysis"
echo "   sbatch scripts/analysis/qualitative_analysis.sh"
echo ""
echo "   # Model comparison analysis"
echo "   sbatch scripts/analysis/model_comparison.sh"
echo ""
echo "3. Individual model training:"
echo "   # Supervised ViT"
echo "   sbatch scripts/vit/train_vit_job.sh"
echo ""
echo "   # DINO SSL"
echo "   sbatch scripts/dino/dino_fine_tune.sh"
echo ""
echo "   # MAE SSL"
echo "   sbatch scripts/mae/run_mae_finetune.sh"
echo ""
echo "   # iBOT SSL"
echo "   sbatch scripts/ibot/run_improved_pipeline.sh"
echo ""
echo "📚 Documentation:"
echo "- README.md: Complete project documentation"
echo "- results/final_evaluation/: Final results and graphs"
echo "- results/qualitative_analysis/: Qualitative analysis results"
echo ""
echo "🔧 External repositories:"
echo "- iBOT: external/ibot/"
echo "- DINO v2: external/dino/"
echo "- MAE: external/mae/"
echo ""
print_success "Happy training! 🚀" 