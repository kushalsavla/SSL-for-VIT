#!/usr/bin/env python3
"""
Master SSL Experiment Runner
Runs experiments for all SSL methods (ViT, MAE, DINO, iBOT).
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_vit_experiments(args):
    """Run ViT baseline experiments."""
    print("🚀 Running ViT baseline experiments...")
    
    if args.mode == 'pretrain' or args.mode == 'all':
        print("📚 Running ViT supervised pretraining...")
        subprocess.run([
            "python", "vit/vit_pretrain.py",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr)
        ], cwd="scripts/experiments")
    
    if args.mode == 'linear_probe' or args.mode == 'all':
        print("🔍 Running ViT linear probing...")
        subprocess.run([
            "python", "vit/vit_linear_probe.py",
            "--epochs", str(args.epochs),
            "--lr", str(args.lr)
        ], cwd="scripts/experiments")
    
    if args.mode == 'nonlinear_probe' or args.mode == 'all':
        print("🔍 Running ViT nonlinear probing...")
        subprocess.run([
            "python", "vit/vit_nonlinear_probe.py",
            "--epochs", str(args.epochs),
            "--lr", str(args.lr)
        ], cwd="scripts/experiments")

def run_mae_experiments(args):
    """Run MAE experiments."""
    print("🚀 Running MAE experiments...")
    
    subprocess.run([
        "python", "mae/run_mae_experiments.py",
        "--mode", args.mode,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size)
    ], cwd="scripts/experiments")

def run_dino_experiments(args):
    """Run DINO experiments."""
    print("🚀 Running DINO experiments...")
    
    subprocess.run([
        "python", "dino/run_dino_experiments.py",
        "--mode", args.mode,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size)
    ], cwd="scripts/experiments")

def run_ibot_experiments(args):
    """Run iBOT experiments."""
    print("🚀 Running iBOT experiments...")
    
    subprocess.run([
        "python", "ibot/run_ibot_experiments.py",
        "--mode", args.mode,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size)
    ], cwd="scripts/experiments")

def main():
    parser = argparse.ArgumentParser(description='Run all SSL experiments')
    parser.add_argument('--methods', nargs='+', 
                       choices=['vit', 'mae', 'dino', 'ibot', 'all'],
                       default=['all'], help='Which SSL methods to run')
    parser.add_argument('--mode', choices=['pretrain', 'linear_probe', 'nonlinear_probe', 'finetune', 'all'], 
                       default='all', help='Which experiments to run')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    print("🚀 Master SSL Experiment Runner")
    print(f"📊 Methods: {args.methods}")
    print(f"📊 Mode: {args.mode}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📚 Epochs: {args.epochs}")
    print(f"📚 Learning rate: {args.lr}")
    
    # Run experiments for each method
    if 'all' in args.methods or 'vit' in args.methods:
        run_vit_experiments(args)
    
    if 'all' in args.methods or 'mae' in args.methods:
        run_mae_experiments(args)
    
    if 'all' in args.methods or 'dino' in args.methods:
        run_dino_experiments(args)
    
    if 'all' in args.methods or 'ibot' in args.methods:
        run_ibot_experiments(args)
    
    print("✅ All experiments completed!")
    print("📊 Check the results/ directory for experiment outputs")

if __name__ == "__main__":
    main()
