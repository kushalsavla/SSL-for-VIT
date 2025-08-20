#!/usr/bin/env python3
"""
Quick test script for iBOT classification
Tests if the classification script can load models and run without errors.
"""

import os
import sys
import torch

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ibot_models():
    """Test if iBOT models can be loaded."""
    print("🧪 Testing iBOT model loading...")
    
    models_dir = "models/ibot"
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return False
    
    # List available models
    print(f"📁 Available models in {models_dir}:")
    for file in os.listdir(models_dir):
        file_path = os.path.join(models_dir, file)
        if file.endswith('.pth'):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   • {file} ({size_mb:.1f} MB)")
    
    # Test loading the main model
    main_model_path = os.path.join(models_dir, "final_model.pth")
    if os.path.exists(main_model_path):
        print(f"\n✅ Testing main model: {main_model_path}")
        try:
            checkpoint = torch.load(main_model_path, map_location='cpu')
            print(f"   ✅ Model loaded successfully")
            
            # Check checkpoint structure
            if 'student' in checkpoint:
                print(f"   📊 Checkpoint type: iBOT SSL training")
                student_keys = list(checkpoint['student'].keys())
                print(f"   📋 Student keys: {len(student_keys)} keys")
                print(f"   🔑 Sample keys: {student_keys[:5]}")
            elif 'model_state_dict' in checkpoint:
                print(f"   📊 Checkpoint type: Fine-tuned model")
            else:
                print(f"   📊 Checkpoint type: Direct model weights")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            return False
    else:
        print(f"❌ Main model not found: {main_model_path}")
        return False

def test_data_availability():
    """Test if CIFAR-10 data is available."""
    print("\n🧪 Testing data availability...")
    
    data_dir = "data/cifar10_splits"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    required_files = ['train_images.npy', 'train_labels.npy', 
                     'val_images.npy', 'val_labels.npy',
                     'test_images.npy', 'test_labels.npy']
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing data files: {missing_files}")
        return False
    
    print(f"✅ All required data files found in {data_dir}")
    
    # Check data sizes
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"   • {file}: {size_mb:.1f} MB")
    
    return True

def test_imports():
    """Test if all required packages can be imported."""
    print("\n🧪 Testing package imports...")
    
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"   ❌ PyTorch: {e}")
        return False
    
    try:
        import timm
        print(f"   ✅ timm: {timm.__version__}")
    except ImportError as e:
        print(f"   ❌ timm: {e}")
        return False
    
    try:
        import numpy as np
        print(f"   ✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"   ❌ NumPy: {e}")
        return False
    
    try:
        from sklearn.metrics import accuracy_score
        print(f"   ✅ scikit-learn: available")
    except ImportError as e:
        print(f"   ❌ scikit-learn: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 iBOT Classification Test Suite")
    print("="*50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Availability", test_data_availability),
        ("iBOT Models", test_ibot_models)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🚀 Ready to run iBOT classification!")
        print("💡 You can now run:")
        print("   • Linear probing: python scripts/ibot/ibot_classification.py --mode linear_probe")
        print("   • Nonlinear probing: python scripts/ibot/ibot_classification.py --mode nonlinear_probe")
        print("   • Fine-tuning: python scripts/ibot/ibot_classification.py --mode fine_tune")
    else:
        print("\n⚠️ Please fix the failed tests before running classification.")

if __name__ == "__main__":
    main()

