#!/usr/bin/env python3
"""
Comprehensive SSL Comparative Study
Compares DINO, MAE, and iBOT with linear and nonlinear probing on CIFAR-10.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
import argparse

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_ssl_comparison(ssl_method, mode, model_path, epochs=100, batch_size=32, lr=1e-4):
    """Run SSL comparison for a specific method and mode."""
    print(f"\n🚀 Running {ssl_method.upper()} with {mode.replace('_', ' ')}...")
    
    if ssl_method == 'ibot':
        # Use the iBOT classification script
        cmd = [
            'python', 'scripts/ibot/ibot_classification.py',
            '--mode', mode,
            '--model_path', model_path,
            '--epochs', str(epochs),
            '--batch_size', str(batch_size),
            '--lr', str(lr),
            '--augmentation', 'medium'
        ]
    elif ssl_method == 'dino':
        # Use DINO scripts
        if mode == 'linear_probe':
            cmd = ['bash', 'scripts/dino/linear_probe.sh']
        elif mode == 'nonlinear_probe':
            cmd = ['bash', 'scripts/dino/nonlinear_probe.sh']
        else:
            print(f"❌ DINO doesn't support {mode} mode")
            return None
    elif ssl_method == 'mae':
        # Use MAE scripts
        if mode == 'linear_probe':
            cmd = ['bash', 'scripts/mae/linear_probe.sh']
        elif mode == 'nonlinear_probe':
            cmd = ['bash', 'scripts/mae/nonlinear_probe.sh']
        else:
            print(f"❌ MAE doesn't support {mode} mode")
            return None
    else:
        print(f"❌ Unknown SSL method: {ssl_method}")
        return None
    
    print(f"🔧 Command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        
        if result.returncode == 0:
            print(f"✅ {ssl_method.upper()} {mode} completed successfully")
            return result.stdout
        else:
            print(f"❌ {ssl_method.upper()} {mode} failed with error:")
            print(result.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {ssl_method.upper()} {mode} timed out after 2 hours")
        return None
    except Exception as e:
        print(f"❌ Error running {ssl_method.upper()} {mode}: {e}")
        return None

def collect_results():
    """Collect results from all completed experiments."""
    results = {}
    
    # Check iBOT results
    ibot_results_dir = Path('results/ibot_experiments')
    if ibot_results_dir.exists():
        for result_file in ibot_results_dir.glob('*_results.json'):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    method = 'ibot'
                    mode = data['mode']
                    key = f"{method}_{mode}"
                    results[key] = {
                        'method': method,
                        'mode': mode,
                        'best_val_acc': data.get('best_val_acc', 0),
                        'test_acc': data.get('test_acc', 0),
                        'epochs': data.get('epochs', 0),
                        'trainable_params': data.get('trainable_params', 0),
                        'total_params': data.get('total_params', 0)
                    }
            except Exception as e:
                print(f"⚠️ Could not read {result_file}: {e}")
    
    # Check DINO results
    dino_results_dir = Path('results/dino_experiments')
    if dino_results_dir.exists():
        for result_file in dino_results_dir.glob('*_results.json'):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    method = 'dino'
                    mode = data.get('mode', 'unknown')
                    key = f"{method}_{mode}"
                    results[key] = {
                        'method': method,
                        'mode': mode,
                        'best_val_acc': data.get('best_val_acc', 0),
                        'test_acc': data.get('test_acc', 0),
                        'epochs': data.get('epochs', 0),
                        'trainable_params': data.get('trainable_params', 0),
                        'total_params': data.get('total_params', 0)
                    }
            except Exception as e:
                print(f"⚠️ Could not read {result_file}: {e}")
    
    # Check MAE results
    mae_results_dir = Path('results/mae_experiments')
    if mae_results_dir.exists():
        for result_file in mae_results_dir.glob('*_results.json'):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    method = 'mae'
                    mode = data.get('mode', 'unknown')
                    key = f"{method}_{mode}"
                    results[key] = {
                        'method': method,
                        'mode': mode,
                        'best_val_acc': data.get('best_val_acc', 0),
                        'test_acc': data.get('test_acc', 0),
                        'epochs': data.get('epochs', 0),
                        'trainable_params': data.get('trainable_params', 0),
                        'total_params': data.get('total_params', 0)
                    }
            except Exception as e:
                print(f"⚠️ Could not read {result_file}: {e}")
    
    return results

def print_comparison_table(results):
    """Print a formatted comparison table."""
    print("\n" + "="*100)
    print("📊 COMPREHENSIVE SSL COMPARISON RESULTS")
    print("="*100)
    
    # Group by mode
    modes = ['linear_probe', 'nonlinear_probe']
    methods = ['ibot', 'dino', 'mae']
    
    for mode in modes:
        print(f"\n🔍 {mode.replace('_', ' ').upper()} RESULTS:")
        print("-" * 80)
        print(f"{'Method':<15} {'Val Acc':<10} {'Test Acc':<10} {'Epochs':<8} {'Trainable Params':<18}")
        print("-" * 80)
        
        for method in methods:
            key = f"{method}_{mode}"
            if key in results:
                data = results[key]
                print(f"{method.upper():<15} {data['best_val_acc']:<10.2f} {data['test_acc']:<10.2f} "
                      f"{data['epochs']:<8} {data['trainable_params']:<18,}")
            else:
                print(f"{method.upper():<15} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<18}")
    
    # Summary statistics
    print("\n" + "="*100)
    print("📈 SUMMARY STATISTICS")
    print("="*100)
    
    for mode in modes:
        print(f"\n🎯 {mode.replace('_', ' ').upper()}:")
        mode_results = {k: v for k, v in results.items() if k.endswith(mode)}
        
        if mode_results:
            best_method = max(mode_results.items(), key=lambda x: x[1]['test_acc'])
            print(f"   🏆 Best: {best_method[1]['method'].upper()} ({best_method[1]['test_acc']:.2f}%)")
            
            avg_acc = sum(r['test_acc'] for r in mode_results.values()) / len(mode_results)
            print(f"   📊 Average: {avg_acc:.2f}%")
        else:
            print(f"   ⚠️ No results available")

def save_comparison_report(results):
    """Save the comparison report to a JSON file."""
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results,
        'summary': {}
    }
    
    # Calculate summary statistics
    modes = ['linear_probe', 'nonlinear_probe']
    for mode in modes:
        mode_results = {k: v for k, v in results.items() if k.endswith(mode)}
        if mode_results:
            best_method = max(mode_results.items(), key=lambda x: x[1]['test_acc'])
            avg_acc = sum(r['test_acc'] for r in mode_results.values()) / len(mode_results)
            
            report['summary'][mode] = {
                'best_method': best_method[1]['method'],
                'best_accuracy': best_method[1]['test_acc'],
                'average_accuracy': avg_acc,
                'num_methods': len(mode_results)
            }
    
    # Save report
    report_path = 'results/ssl_comparison_report.json'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Comparison report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive SSL Comparative Study')
    parser.add_argument('--run_experiments', action='store_true', 
                       help='Run all SSL experiments (linear and nonlinear probing)')
    parser.add_argument('--collect_results', action='store_true',
                       help='Collect and display results from completed experiments')
    parser.add_argument('--methods', nargs='+', choices=['ibot', 'dino', 'mae'], 
                       default=['ibot', 'dino', 'mae'],
                       help='SSL methods to compare')
    parser.add_argument('--modes', nargs='+', choices=['linear_probe', 'nonlinear_probe'],
                       default=['linear_probe', 'nonlinear_probe'],
                       help='Probing modes to run')
    
    args = parser.parse_args()
    
    if not args.run_experiments and not args.collect_results:
        print("❌ Please specify either --run_experiments or --collect_results")
        return
    
    if args.run_experiments:
        print("🚀 Starting Comprehensive SSL Comparative Study")
        print("="*60)
        print("📊 This will run:")
        for method in args.methods:
            print(f"   • {method.upper()}: Linear & Nonlinear probing")
        print("="*60)
        
        # Model paths for each method
        model_paths = {
            'ibot': 'models/ibot/final_model.pth',
            'dino': 'models/dino/best_model.pth',  # Adjust as needed
            'mae': 'models/mae/best_model.pth'     # Adjust as needed
        }
        
        # Run experiments
        for method in args.methods:
            if method not in model_paths:
                print(f"⚠️ No model path configured for {method}, skipping...")
                continue
                
            model_path = model_paths[method]
            if not os.path.exists(model_path):
                print(f"⚠️ Model not found for {method} at {model_path}, skipping...")
                continue
            
            for mode in args.modes:
                print(f"\n{'='*60}")
                print(f"🚀 Running {method.upper()} - {mode.replace('_', ' ').title()}")
                print(f"{'='*60}")
                
                run_ssl_comparison(method, mode, model_path)
                
                # Wait a bit between experiments
                time.sleep(5)
        
        print("\n✅ All experiments completed!")
    
    if args.collect_results:
        print("📊 Collecting results from completed experiments...")
        results = collect_results()
        
        if results:
            print_comparison_table(results)
            save_comparison_report(results)
        else:
            print("⚠️ No results found. Run experiments first with --run_experiments")

if __name__ == "__main__":
    main()

