#!/usr/bin/env python3
"""
Performance Visualization for SSL ViT Pipeline
Generates comparison plots and analysis visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_performance_comparison():
    """Create main performance comparison plot"""
    
    # Results data
    methods = ['Linear Probing', 'Non-linear Probing', 'Fine-tuning']
    accuracies = [15.65, 22.96, 68.73]
    improvements = [0, 7.31, 53.08]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('SSL Pipeline Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 80)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Improvement comparison
    bars2 = ax2.bar(methods, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Improvement over Linear Probing (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Improvement Analysis', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 60)
    
    # Add value labels on bars
    for bar, imp in zip(bars2, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'+{imp:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_class_performance_heatmap():
    """Create class-specific performance heatmap"""
    
    # Fine-tuning class results
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    precision = [0.7179, 0.7471, 0.5998, 0.5306, 0.6701, 
                 0.5853, 0.7333, 0.7389, 0.7873, 0.7405]
    
    recall = [0.7150, 0.7830, 0.5710, 0.4850, 0.6480, 
              0.6070, 0.8030, 0.7020, 0.8030, 0.7560]
    
    f1_score = [0.7164, 0.7646, 0.5850, 0.5068, 0.6589, 
                0.5960, 0.7666, 0.7200, 0.7950, 0.7481]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    })
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    metrics_data = df[['Precision', 'Recall', 'F1-Score']].values
    sns.heatmap(metrics_data.T, 
                xticklabels=classes,
                yticklabels=['Precision', 'Recall', 'F1-Score'],
                annot=True, 
                fmt='.3f',
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Score'},
                square=True)
    
    plt.title('Fine-tuning: Class-Specific Performance Metrics', fontsize=14, fontweight='bold')
    plt.xlabel('CIFAR-10 Classes', fontsize=12, fontweight='bold')
    plt.ylabel('Metrics', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('class_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_ssl_training_progression():
    """Create SSL pre-training loss progression plot"""
    
    # Loss progression data (from the training logs)
    epochs = [1, 23, 36, 59, 63, 66, 100]
    losses = [8.0, 4.8, 2.72, 0.8326, 0.5384, 0.3993, 0.3993]
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, losses, 'o-', linewidth=3, markersize=8, color='#FF6B6B')
    plt.fill_between(epochs, losses, alpha=0.3, color='#FF6B6B')
    
    plt.xlabel('Training Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('SSL Loss', fontsize=12, fontweight='bold')
    plt.title('SSL Pre-training: Loss Progression', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add annotations for key points
    plt.annotate('Initial Loss: 8.0', xy=(1, 8.0), xytext=(10, 7.0),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, fontweight='bold')
    
    plt.annotate('Converged: 0.3993', xy=(100, 0.3993), xytext=(70, 1.5),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ssl_training_progression.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_computational_efficiency_chart():
    """Create computational efficiency comparison"""
    
    methods = ['Linear Probing', 'Non-linear Probing', 'Fine-tuning']
    training_times = [2, 5, 48]  # minutes
    accuracies = [15.65, 22.96, 68.73]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot
    scatter = ax.scatter(training_times, accuracies, 
                        s=[200, 200, 200], 
                        c=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                        alpha=0.8, edgecolors='black', linewidth=2)
    
    # Add labels
    for i, method in enumerate(methods):
        ax.annotate(method, (training_times[i], accuracies[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Computational Efficiency vs Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('computational_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_dashboard():
    """Create a comprehensive summary dashboard"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Performance comparison
    ax1 = fig.add_subplot(gs[0, 0])
    methods = ['Linear\nProbing', 'Non-linear\nProbing', 'Fine-tuning']
    accuracies = [15.65, 22.96, 68.73]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax1.bar(methods, accuracies, color=colors, alpha=0.8)
    ax1.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax1.set_title('Performance Comparison', fontweight='bold')
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Training time comparison
    ax2 = fig.add_subplot(gs[0, 1])
    times = [2, 5, 48]
    bars = ax2.bar(methods, times, color=colors, alpha=0.8)
    ax2.set_ylabel('Training Time (min)', fontweight='bold')
    ax2.set_title('Computational Cost', fontweight='bold')
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{time}min', ha='center', va='bottom', fontweight='bold')
    
    # 3. Improvement analysis
    ax3 = fig.add_subplot(gs[0, 2])
    improvements = [0, 7.31, 53.08]
    bars = ax3.bar(methods, improvements, color=colors, alpha=0.8)
    ax3.set_ylabel('Improvement (%)', fontweight='bold')
    ax3.set_title('Improvement over Linear Probing', fontweight='bold')
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'+{imp:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. SSL training progression
    ax4 = fig.add_subplot(gs[1, :2])
    epochs = [1, 23, 36, 59, 63, 66, 100]
    losses = [8.0, 4.8, 2.72, 0.8326, 0.5384, 0.3993, 0.3993]
    ax4.plot(epochs, losses, 'o-', linewidth=3, markersize=8, color='#FF6B6B')
    ax4.fill_between(epochs, losses, alpha=0.3, color='#FF6B6B')
    ax4.set_xlabel('Training Epochs', fontweight='bold')
    ax4.set_ylabel('SSL Loss', fontweight='bold')
    ax4.set_title('SSL Pre-training: Loss Progression', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Class performance summary
    ax5 = fig.add_subplot(gs[1, 2])
    classes = ['airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    f1_scores = [0.7164, 0.7646, 0.5850, 0.5068, 0.6589, 0.5960, 0.7666, 0.7200, 0.7950, 0.7481]
    bars = ax5.barh(classes, f1_scores, color='#45B7D1', alpha=0.8)
    ax5.set_xlabel('F1-Score', fontweight='bold')
    ax5.set_title('Fine-tuning: Class Performance', fontweight='bold')
    ax5.set_xlim(0, 1)
    
    plt.suptitle('SSL for Vision Transformers: Complete Pipeline Summary', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('ssl_pipeline_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("🎨 Generating SSL Pipeline Performance Visualizations...")
    
    # Create all visualizations
    create_performance_comparison()
    create_class_performance_heatmap()
    create_ssl_training_progression()
    create_computational_efficiency_chart()
    create_summary_dashboard()
    
    print("✅ All visualizations generated successfully!")
    print("📊 Generated files:")
    print("  - performance_comparison.png")
    print("  - class_performance_heatmap.png")
    print("  - ssl_training_progression.png")
    print("  - computational_efficiency.png")
    print("  - ssl_pipeline_dashboard.png") 