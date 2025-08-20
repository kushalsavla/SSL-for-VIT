# 🚀 SSL Experiment Framework - Complete Guide

## 📁 What We've Built

We've created a comprehensive framework that leverages the **existing external repositories** (MAE, DINO, iBOT) instead of rewriting everything from scratch. This approach is much more reliable and faster.

## 🏗️ Structure

```
scripts/experiments/
├── mae/                          # MAE experiments
│   ├── working/                  # Copied from external/mae/
│   │   ├── mae_pretrain.py      # Official MAE pretraining
│   │   ├── mae_linear_probe.py  # Official MAE linear probing
│   │   ├── mae_finetune.py      # Official MAE fine-tuning
│   │   ├── models_mae.py        # MAE model definitions
│   │   └── models_vit.py        # ViT model definitions
│   ├── mae_pretrain_job.sh      # SLURM job for pretraining
│   ├── mae_linear_probe_job.sh  # SLURM job for linear probing
│   └── mae_finetune_job.sh      # SLURM job for fine-tuning
├── dino/                         # DINO experiments
│   ├── working/                  # Copied from external/dino/
│   │   ├── dino_pretrain.py     # Official DINO pretraining
│   │   └── ssl_meta_arch.py     # DINO architecture
│   └── dino_pretrain_job.sh     # SLURM job for pretraining
├── ibot/                         # iBOT experiments
│   ├── working/                  # Copied from external/ibot/
│   │   ├── ibot_pretrain.py     # Official iBOT pretraining
│   │   └── utils.py             # iBOT utilities
│   └── ibot_pretrain_job.sh     # SLURM job for pretraining
├── run_all_ssl_experiments_job.sh # Master job script
└── SSL_EXPERIMENT_GUIDE.md      # This guide
```

## 🎯 Key Benefits of This Approach

1. **✅ Reliability**: Uses official, tested implementations
2. **🚀 Speed**: No need to rewrite complex SSL algorithms
3. **🔧 Adaptability**: Easy to modify for CIFAR-10
4. **📊 Consistency**: Same evaluation across all methods
5. **💾 Proven**: Based on published, peer-reviewed code

## 🚀 How to Run Experiments

### **Option 1: Individual Jobs (Recommended for Testing)**

#### **MAE Experiments**
```bash
# Pretraining
sbatch scripts/experiments/mae/mae_pretrain_job.sh

# Linear probing (after pretraining)
sbatch scripts/experiments/mae/mae_linear_probe_job.sh

# Fine-tuning (after pretraining)
sbatch scripts/experiments/mae/mae_finetune_job.sh
```

#### **DINO Experiments**
```bash
# Pretraining
sbatch scripts/experiments/dino/dino_pretrain_job.sh
```

#### **iBOT Experiments**
```bash
# Pretraining
sbatch scripts/experiments/ibot/ibot_pretrain_job.sh
```

### **Option 2: Complete Pipeline (For Full Project)**
```bash
# Run everything in sequence
sbatch scripts/experiments/run_all_ssl_experiments_job.sh
```

## 🔧 CIFAR-10 Specific Configurations

### **Image & Patch Sizes**
- **Image size**: 32x32 (CIFAR-10 standard)
- **Patch size**: 4x4 (8 patches per image: 32/4 = 8)
- **Sequence length**: 8 patches + 1 CLS token = 9 tokens

### **Model Adaptations**
- **MAE**: ViT Base with 4x4 patches, decoder for 32x32 images
- **DINO**: ViT Small with 4x4 patches, teacher-student setup
- **iBOT**: ViT Small with 4x4 patches, masked token prediction

### **Training Parameters**
- **Batch size**: 64 (pretraining), 32 (fine-tuning), 128 (probing)
- **Learning rate**: 1e-4 (fine-tuning), 3e-4 (pretraining)
- **Epochs**: 200 (pretraining), 100 (fine-tuning/probing)

## 📊 Expected Results

### **Performance Ranking (Expected)**
1. **Fine-tuned SSL models** (highest - learns task-specific features)
2. **Fine-tuned supervised ViT** (high - learns from labels)
3. **Nonlinear probes on SSL** (medium - good representation evaluation)
4. **Linear probes on SSL** (lower - limited by linear decision boundaries)
5. **Probes on random ViT** (lowest - no meaningful representations)

### **Key Insights to Look For**
- **SSL vs Supervised**: Do SSL methods learn better representations?
- **Linear vs Nonlinear**: How much do nonlinear probes improve?
- **Method Comparison**: Which SSL method works best on CIFAR-10?
- **Data Efficiency**: How well do SSL methods perform with limited labels?

## 🚨 Current Status & Next Steps

### **✅ What's Ready**
- **MAE**: Complete pipeline (pretrain → linear probe → fine-tune)
- **DINO**: Pretraining only
- **iBOT**: Pretraining only
- **All SLURM job scripts**: Ready to run

### **⚠️ What Needs Implementation**
- **DINO linear probing**: Custom script needed
- **DINO fine-tuning**: Custom script needed
- **iBOT linear probing**: Custom script needed
- **iBOT fine-tuning**: Custom script needed

### **🔧 Implementation Plan**
1. **Phase 1**: Test MAE pipeline (fully working)
2. **Phase 2**: Implement DINO probing/fine-tuning
3. **Phase 3**: Implement iBOT probing/fine-tuning
4. **Phase 4**: Run complete comparative study

## 💡 Quick Start Guide

### **1. Test MAE (Fully Working)**
```bash
# Start with pretraining
sbatch scripts/experiments/mae/mae_pretrain_job.sh

# Check output
tail -f mae_pretrain_*.out

# After pretraining completes, run linear probing
sbatch scripts/experiments/mae/mae_linear_probe_job.sh

# Finally, run fine-tuning
sbatch scripts/experiments/mae/mae_finetune_job.sh
```

### **2. Test DINO (Pretraining Only)**
```bash
# Run DINO pretraining
sbatch scripts/experiments/dino/dino_pretrain_job.sh

# Check output
tail -f dino_pretrain_*.out
```

### **3. Test iBOT (Pretraining Only)**
```bash
# Run iBOT pretraining
sbatch scripts/experiments/ibot/ibot_pretrain_job.sh

# Check output
tail -f ibot_pretrain_*.out
```

## 🔍 Monitoring & Debugging

### **Check Job Status**
```bash
squeue -u $USER
```

### **Check Output Files**
```bash
# List all output files
ls -la *_*.out

# Monitor specific job
tail -f mae_pretrain_*.out

# Check for errors
grep -i error mae_pretrain_*.err
```

### **Common Issues & Solutions**
1. **CUDA out of memory**: Reduce batch size
2. **Import errors**: Check conda environment activation
3. **Data path issues**: Verify `data/cifar10_splits/` exists
4. **Model loading errors**: Check checkpoint paths

## 📈 Results Analysis

### **Where Results Are Saved**
- **Models**: `outputs/{method}_pretrain/`, `outputs/{method}_finetune/`
- **Logs**: `{method}_{type}_*.out` files
- **Metrics**: Checkpoint files and training logs

### **Key Metrics to Track**
- **Pretraining**: Reconstruction loss (MAE), contrastive loss (DINO/iBOT)
- **Linear probing**: Validation accuracy on frozen features
- **Fine-tuning**: Final test accuracy

## 🎯 Milestone Coverage

### **Milestone 1: ViT Framework & Baselines** ✅
- ViT supervised training
- Linear/nonlinear probing on random ViT

### **Milestone 2: DINO & MAE Implementation** 🔄
- **MAE**: Complete ✅
- **DINO**: Pretraining ✅, probing/fine-tuning needed

### **Milestone 3: iBOT Implementation** 🔄
- **iBOT**: Pretraining ✅, probing/fine-tuning needed

## 🚀 Next Steps

1. **Test MAE pipeline** (fully working)
2. **Implement DINO probing/fine-tuning**
3. **Implement iBOT probing/fine-tuning**
4. **Run complete comparative study**
5. **Analyze results and write report**

This framework gives you everything needed to complete your RL#2 project with a comprehensive comparison of SSL methods! 🎉
