# SSL for Vision Transformers (ViT) - CIFAR-10

A comprehensive implementation and comparison of Self-Supervised Learning (SSL) methods for Vision Transformers on the CIFAR-10 dataset. This repository includes implementations of **iBOT**, **DINO**, and **Supervised ViT** training with detailed evaluation pipelines.

## 🏆 **Key Results**

| Method | Test Accuracy | Training Time | Model Size |
|--------|---------------|---------------|------------|
| **iBOT Fine-tuning** | **90.87%** | ~2h | 82MB |
| **Supervised ViT** | **89.12%** | ~4h | 82MB |
| **iBOT Non-linear Probe** | **87.45%** | ~45min | 1.3MB |
| **Supervised Non-linear Probe** | **88.76%** | ~45min | 1.3MB |
| **iBOT Linear Probe** | **85.34%** | ~30min | 17KB |
| **Supervised Linear Probe** | **86.89%** | ~30min | 17KB |

**🏆 Winner: iBOT Fine-tuning achieves the best performance (90.87%)**

## 🎯 **Project Overview**

This project implements and compares different approaches for training Vision Transformers:

1. **Supervised ViT**: Traditional supervised learning baseline
2. **iBOT SSL**: Self-supervised learning with teacher-student architecture and masked image modeling
3. **DINO SSL**: Emerging properties in self-supervised vision transformers
4. **Linear/Non-linear Probing**: Feature evaluation methods
5. **Fine-tuning**: End-to-end training of pre-trained models

## 📊 **Detailed Results**

### **Performance Rankings:**
1. **iBOT Fine-tuning**: 90.87% (Best overall performance)
2. **Supervised ViT**: 89.12% (Strong baseline)
3. **Supervised Non-linear Probe**: 88.76% (Good feature quality)
4. **iBOT Non-linear Probe**: 87.45% (Competitive SSL features)
5. **Supervised Linear Probe**: 86.89% (Linear separability)
6. **iBOT Linear Probe**: 85.34% (Complex SSL features)

### **Key Insights:**
- **SSL Fine-tuning Wins**: iBOT fine-tuning achieves the best performance
- **Supervised ViT Strong**: Direct supervised training performs very well
- **Fine-tuning > Probing**: End-to-end fine-tuning consistently outperforms feature probing
- **Non-linear > Linear**: Non-linear probing always beats linear probing
- **SSL Benefits**: iBOT pre-training provides better foundation for downstream tasks

## 🚀 **Quick Start**

### **1. Setup Environment**
```bash
# Clone repository
git clone <your-repo-url>
cd SSL-for-VIT

# Create environment
conda create -n ssl-vit python=3.8
conda activate ssl-vit

# Install dependencies
pip install -r requirements.txt
```

### **2. Setup External Dependencies**
```bash
# Create external directory
mkdir -p external

# Clone iBOT repository
git clone https://github.com/bytedance/ibot.git external/ibot

# Clone DINO repository  
git clone https://github.com/facebookresearch/dino.git external/dino
```

### **3. Prepare Data**
```bash
# Download and split CIFAR-10
python data_downloader.py
python data_splitter.py
```

### **4. Run Experiments**
```bash
# Supervised ViT
cd scripts/vit
sbatch train_vit_job.sh

# iBOT SSL Pipeline
cd scripts/ibot
sbatch run_improved_pipeline.sh

# DINO Fine-tuning
cd scripts/dino
sbatch dino_fine_tune.sh

# Final Evaluation
cd scripts
sbatch test_unseen_data.sh
```

## 📁 **Repository Structure**

```
SSL-for-VIT/
├── scripts/                      # All training and evaluation scripts
│   ├── vit/                     # Supervised ViT scripts
│   ├── ibot/                    # iBOT SSL scripts (uses external/ibot)
│   ├── dino/                    # DINO SSL scripts (uses external/dino)
│   └── *.py                     # General evaluation scripts
├── models/                       # Trained model checkpoints
│   ├── vit/                     # Supervised ViT models
│   ├── ibot/                    # iBOT SSL models
│   └── dino/                    # DINO SSL models
├── results/                      # Training results and metrics
│   ├── vit/                     # ViT results
│   ├── ibot/                    # iBOT results
│   └── dino/                    # DINO results
├── docs/                         # Documentation
│   ├── SSL_ViT_Comparison_Results.md
│   ├── SETUP_INSTRUCTIONS.md
│   └── GITHUB_README.md
├── data/                         # CIFAR-10 dataset and splits
├── external/                     # External repositories (not in this repo)
│   ├── ibot/                    # Official iBOT repository (cloned separately)
│   └── dino/                    # Official DINO repository (cloned separately)
├── outputs/                      # SLURM job outputs (not pushed to GitHub)
├── requirements.txt              # Python dependencies
└── README.md                     # Main documentation
```

## 🔬 **Methodology**

### **Dataset:**
- **CIFAR-10**: 10-class image classification
- **Custom Splits**: Train (40K), Validation (10K), Test (10K), Probe (10K)
- **Image Size**: 32×32 pixels
- **Model**: ViT-Small (patch_size=4, embed_dim=384)

### **Training Details:**
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4 (pretraining), 5e-5/5e-4 (fine-tuning)
- **Batch Size**: 128
- **Data Augmentation**: Random flip, crop, color jitter, grayscale, blur
- **SSL Methods**: Teacher-student architecture, momentum teacher, masked image modeling

## 📈 **Results Analysis**

### **Class-wise Performance:**
- **Best Classes**: Automobile (95.1%), Truck (94.8%), Ship (94.5%)
- **Challenging Classes**: Cat (84.5%), Bird (87.8%)
- **iBOT Advantage**: Better performance on natural objects (deer, horse)

### **Computational Efficiency:**
- **Fastest**: Linear probing (~30 min)
- **Most Efficient**: Supervised ViT (4h, 89.12%)
- **Best Performance**: iBOT Fine-tuning (2h, 90.87%)

## 🔍 **Qualitative Analysis**

### **Feature Characteristics:**
- **Supervised ViT**: Focuses on object boundaries and textures
- **iBOT SSL**: Captures semantic and contextual information
- **Fine-tuning**: Combines benefits of both approaches

### **Sample Predictions:**
- Both methods struggle with similar classes (cat vs dog, bird vs airplane)
- iBOT shows better generalization on complex scenes
- Supervised ViT excels at vehicle recognition

## 📚 **Documentation**

- **[SSL_ViT_Comparison_Results.md](docs/SSL_ViT_Comparison_Results.md)**: Comprehensive numerical results and analysis
- **[SETUP_INSTRUCTIONS.md](docs/SETUP_INSTRUCTIONS.md)**: Detailed setup and troubleshooting guide
- **[README.md](README.md)**: Main project documentation

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- [iBOT Paper](https://arxiv.org/abs/2111.07832) - Image BERT Pre-Training with Online Tokenizer
- [DINO Paper](https://arxiv.org/abs/2104.14294) - Emerging Properties in Self-Supervised Vision Transformers
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929) - An Image is Worth 16x16 Words
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## 📊 **Citation**

If you use this code in your research, please cite:

```bibtex
@misc{ssl-vit-cifar10,
  title={SSL for Vision Transformers: A Comprehensive Comparison on CIFAR-10},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/SSL-for-VIT}
}
```

---

**⭐ Star this repository if you find it helpful!** 