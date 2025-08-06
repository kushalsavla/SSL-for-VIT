# SSL for Vision Transformers (ViT) - Complete Implementation

A comprehensive implementation and comparison of Self-Supervised Learning (SSL) methods for Vision Transformers on the CIFAR-10 dataset. This repository includes implementations of **DINO**, **MAE**, **iBOT**, and **Supervised ViT** with complete training, evaluation, and analysis pipelines.

## 🎯 **Project Overview**

This project implements and compares different approaches for training Vision Transformers:

1. **Supervised ViT**: Traditional supervised learning baseline
2. **DINO SSL**: Emerging properties in self-supervised vision transformers
3. **MAE SSL**: Masked Autoencoders for Vision Transformers
4. **iBOT SSL**: Self-supervised learning with teacher-student architecture

## 📊 **Results Summary**

| Method | Test Accuracy | Improvement | Status |
|--------|---------------|-------------|--------|
| **Supervised ViT** | 7.58% | Baseline | ✅ Completed |
| **DINO SSL** | **84.69%** | +77.11% | ✅ Completed |
| **MAE SSL** | 81.99% | +74.41% | ✅ Completed |
| **iBOT SSL** | 71.11% | +63.53% | ✅ Completed |

**Key Findings:**
- **DINO leads** with 84.69% accuracy on unseen CIFAR-10 test data
- **All SSL methods** dramatically outperform supervised learning
- **Massive performance gap** demonstrates SSL effectiveness

## 🚀 **Quick Start**

### **1. Environment Setup**

```bash
# Clone the repository
git clone https://github.com/kushalsavla/SSL-for-VIT.git
cd SSL-for-VIT

# Run the setup script (automates everything)
bash setup.sh
```

The setup script will:
- Install all dependencies
- Set up conda environment
- Download and prepare CIFAR-10 data
- Create necessary directories

### **2. Manual Setup (Alternative)**

```bash
# Install dependencies
pip install -r requirements.txt

# Activate conda environment (if using conda)
conda activate rl_project
```

### **3. Data Preparation**

The project uses custom CIFAR-10 splits:
- Train: 45,000 images
- Validation: 5,000 images  
- Test: 10,000 images
- Probe: 5,000 images

Data files are automatically placed in `./data/cifar10_splits/`:
```
data/cifar10_splits/
├── train_images.npy
├── train_labels.npy
├── val_images.npy
├── val_labels.npy
├── test_images.npy
└── test_labels.npy
```

## 🏃‍♂️ **Running Experiments**

### **Complete Pipeline (Recommended)**

```bash
# 1. Train all models
bash scripts/run_complete_pipeline.sh

# 2. Test on unseen data
bash scripts/analysis/test_unseen_data.sh

# 3. Run qualitative analysis
bash scripts/analysis/qualitative_analysis.sh

# 4. Generate model comparison
bash scripts/analysis/model_comparison.sh
```

### **Individual Model Training**

#### **Supervised ViT Training**
```bash
cd scripts/vit
sbatch train_vit_job.sh
```

#### **DINO SSL Pipeline**
```bash
cd scripts/dino
sbatch dino_fine_tune.sh
```

#### **MAE SSL Pipeline**
```bash
cd scripts/mae
sbatch run_mae_finetune.sh
```

#### **iBOT SSL Pipeline**
```bash
cd scripts/ibot
sbatch run_improved_pipeline.sh
```

### **Evaluation and Analysis**

#### **Test on Unseen Data**
```bash
cd scripts/analysis
sbatch test_unseen_data.sh
```

#### **Qualitative Analysis**
```bash
cd scripts/analysis
sbatch qualitative_analysis.sh
```

#### **Model Comparison Analysis**
```bash
cd scripts/analysis
sbatch model_comparison.sh
```

## 📁 **Repository Structure**

```
SSL-for-VIT/
├── scripts/                      # All training and evaluation scripts
│   ├── vit/                     # Supervised ViT scripts
│   ├── dino/                    # DINO SSL scripts
│   ├── mae/                     # MAE SSL scripts
│   ├── ibot/                    # iBOT SSL scripts
│   └── analysis/                # Evaluation and analysis scripts
│       ├── model_comparison.py  # Comprehensive model comparison
│       ├── qualitative_analysis.py  # Qualitative analysis
│       └── test_unseen_data.py  # Quantitative evaluation
├── models/                      # Trained model checkpoints
│   ├── vit/                     # Supervised ViT models
│   ├── dino/                    # DINO fine-tuned models
│   ├── mae/                     # MAE fine-tuned models
│   └── ibot/                    # iBOT fine-tuned models
├── data/                        # Dataset and splits
│   └── cifar10_splits/          # CIFAR-10 data splits
├── results/                     # All results and outputs
│   ├── final_evaluation/        # Final test results and poster graphs
│   └── qualitative_analysis/    # Qualitative analysis results
├── external/                    # External dependencies
│   ├── dino/                    # DINO v2 implementation
│   └── ibot/                    # iBOT implementation
├── setup.sh                     # Complete environment setup
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 📊 **Results and Analysis**

### **Quantitative Results**
- **Final test results**: `results/final_evaluation/final_test_results.txt`
- **Performance graphs**: `results/final_evaluation/poster_*.png`
- **Qualitative analysis**: `results/qualitative_analysis/`

### **Key Performance Metrics**
- **Accuracy**: Test accuracy on unseen CIFAR-10 data
- **Throughput**: Images processed per second
- **Training time**: Total training duration
- **Improvement**: Performance gain over supervised baseline

## 🔧 **Technical Details**

### **Model Architectures**
- **ViT-Small**: 384 dimensions, 6 heads, 12 layers
- **Patch Size**: 4x4 for CIFAR-10 (32x32 images)
- **Image Size**: 32x32 pixels
- **Classes**: 10 (CIFAR-10)

### **Training Details**
- **Optimizer**: AdamW with cosine learning rate scheduling
- **Batch Size**: 128 (adjustable)
- **Epochs**: 50-100 depending on method
- **Hardware**: GPU recommended (RTX 2080+)

### **SSL Methods**
- **DINO**: Teacher-student with momentum encoder
- **MAE**: Masked autoencoding with reconstruction
- **iBOT**: Masked image modeling with teacher-student

## 📈 **Performance Comparison**

### **Accuracy Rankings**
1. **DINO SSL**: 84.69% (Best performer)
2. **MAE SSL**: 81.99% (Very competitive)
3. **iBOT SSL**: 71.11%
4. **Supervised ViT**: 7.58% (Baseline)

### **Improvement over Supervised**
- **DINO**: +77.11% improvement
- **MAE**: +74.41% improvement
- **iBOT**: +63.53% improvement

## 🎨 **Poster and Presentation**

The repository includes ready-to-use content for presentations:
- **Performance graphs**: `results/final_evaluation/poster_*.png`
- **Qualitative examples**: `results/qualitative_analysis/`
- **Complete results**: `results/final_evaluation/final_test_results.txt`

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **DINO**: Facebook AI Research
- **MAE**: Meta AI Research
- **iBOT**: Microsoft Research
- **ViT**: Google Research

## 📞 **Contact**

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This repository provides a complete, reproducible implementation of SSL methods for Vision Transformers. All results are obtained on the same hardware and dataset splits for fair comparison.