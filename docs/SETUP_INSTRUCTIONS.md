# Setup Instructions

## 🚀 **Quick Start**

This repository contains SSL ViT implementations for CIFAR-10. Follow these instructions to set up the environment and run the experiments.

---

## 📋 **Prerequisites**

- Python 3.8+
- CUDA-compatible GPU (recommended)
- SLURM cluster access (for job scripts)
- Git

---

## 🔧 **Environment Setup**

### **1. Clone this repository**
```bash
git clone <your-repo-url>
cd SSL-for-VIT
```

### **2. Create conda environment**
```bash
conda create -n ssl-vit python=3.8
conda activate ssl-vit
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 📚 **External Dependencies**

### **1. iBOT Repository**
The iBOT implementation requires the official iBOT repository.

```bash
# Clone iBOT repository
git clone https://github.com/bytedance/ibot.git external/ibot
cd external/ibot

# Install iBOT dependencies
pip install -r requirements.txt

# Return to main directory
cd ../..
```

**Note**: Our scripts automatically detect and add the iBOT repository to the Python path. If you see a warning about the repository not being found, make sure to clone it to the `external/ibot` directory.

### **2. DINO Repository**
For DINO implementation, clone the official DINO repository:

```bash
# Clone DINO repository
git clone https://github.com/facebookresearch/dino.git external/dino
cd external/dino

# Install DINO dependencies
pip install -r requirements.txt

# Return to main directory
cd ../..
```

**Note**: Update import paths in DINO scripts:
```python
# In scripts/dino/dino_fine_tune.py
import sys
sys.path.append('external/dino')
from models.vision_transformer import vit_small
```

### **3. Vision Transformer Models**
For ViT models, we use the `timm` library:

```bash
pip install timm
```

---

## 📁 **Data Setup**

### **1. Download CIFAR-10**
```bash
python data_downloader.py
```

### **2. Split data into custom splits**
```bash
python data_splitter.py
```

This creates the following structure:
```
data/
├── cifar10_splits/
│   ├── train_images.npy
│   ├── train_labels.npy
│   ├── val_images.npy
│   ├── val_labels.npy
│   ├── test_images.npy
│   ├── test_labels.npy
│   ├── probe_images.npy
│   └── probe_labels.npy
```

---

## 🏃‍♂️ **Running Experiments**

### **1. Supervised ViT Training**
```bash
cd scripts/vit
sbatch train_vit_job.sh
```

### **2. iBOT SSL Pre-training**
```bash
cd scripts/ibot
sbatch ibot_pretrain.sh
```

### **3. Linear Probing**
```bash
# For supervised ViT
cd scripts/vit
sbatch linear_probe_vit.sh

# For iBOT
cd scripts/ibot
sbatch linear_probe.sh
```

### **4. Non-linear Probing**
```bash
# For supervised ViT
cd scripts/vit
sbatch nonlinear_probe_vit.sh

# For iBOT
cd scripts/ibot
sbatch nonlinear_probe.sh
```

### **5. Fine-tuning**
```bash
cd scripts/ibot
sbatch fine_tune.sh
```

### **6. DINO Fine-tuning**
```bash
cd scripts/dino
sbatch dino_fine_tune.sh
```

### **7. Final Evaluation**
```bash
cd scripts
sbatch test_unseen_data.sh
```

### **8. Qualitative Analysis**
```bash
cd scripts
sbatch qualitative_analysis.sh
```

---

## 📊 **Results**

Results are saved in the following structure:
```
results/
├── vit/
│   ├── linear_probe_results/
│   └── nonlinear_probe_results/
├── ibot/
│   ├── linear_probe_results/
│   ├── nonlinear_probe_results/
│   └── fine_tune_results/
└── dino/
    └── fine_tune_results/
```

### **Model Checkpoints**
Trained models are saved in:
```
models/
├── vit/
├── ibot/
└── dino/
```

---

## 🔍 **Troubleshooting**

### **Common Issues:**

1. **Import Errors**: Make sure external repositories are cloned and paths are updated
2. **CUDA Memory**: Reduce batch size in job scripts
3. **SLURM Jobs**: Check job output files for errors
4. **Data Loading**: Ensure CIFAR-10 data is properly downloaded and split

### **Environment Variables:**
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/external/ibot:$(pwd)/external/dino"
```

---

## 📝 **Customization**

### **Modifying Hyperparameters:**
Edit the job scripts in `scripts/` directory to change:
- Learning rates
- Batch sizes
- Number of epochs
- Model architectures

### **Adding New Methods:**
1. Create new script in appropriate `scripts/` subdirectory
2. Update job script with correct parameters
3. Add results to comparison document

---

## 📚 **References**

- [iBOT Paper](https://arxiv.org/abs/2111.07832)
- [DINO Paper](https://arxiv.org/abs/2104.14294)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 