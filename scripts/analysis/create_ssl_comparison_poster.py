#!/usr/bin/env python3
"""
SSL vs Frozen Supervised ViT Comparison Poster
Creates paired bar graphs comparing each SSL method against ViT baseline
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

def create_ssl_comparison_poster():
    """Create the main comparison poster with paired bar graphs"""
    
    # CORRECT DATA from final_test_results.txt
    # Linear Classifier Results
    linear_data = {
        'ViT': 37.04,      # Frozen Supervised ViT baseline
        'DINO': 79.38,     # DINO SSL
        'MAE': 71.34,      # MAE ViT (SSL)
        'iBOT': 68.73      # iBOT SSL
    }
    
    # Non-Linear Classifier Results  
    nonlinear_data = {
        'ViT': 36.92,      # Frozen Supervised ViT baseline
        'DINO': 85.80,     # DINO SSL
        'MAE': 86.66,      # MAE ViT (SSL)
        'iBOT': 71.11      # iBOT SSL
    }
    
    # Create the figure with two subplots - optimized height without extra text
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 12))
    
    # Color scheme
    vit_color = '#6C757D'      # Gray for ViT baseline
    ssl_colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange for SSL methods
    
    # Plot 1: Linear Classifier Paired Comparisons
    methods = ['DINO', 'MAE', 'iBOT']
    x_pos = np.arange(len(methods))
    width = 0.3  # Reduced width for better spacing
    
    # Create paired bars for each comparison
    for i, method in enumerate(methods):
        # ViT baseline bar
        ax1.bar(i - width/2, linear_data['ViT'], width, 
                label='Frozen Supervised ViT' if i == 0 else "", 
                color=vit_color, alpha=0.8, edgecolor='black', linewidth=1)
        
        # SSL method bar
        ax1.bar(i + width/2, linear_data[method], width, 
                label=f'{method} SSL' if i == 0 else "", 
                color=ssl_colors[i], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add improvement text above SSL bars - positioned higher to avoid overlap
        improvement = linear_data[method] - linear_data['ViT']
        if improvement > 0:
            ax1.text(i + width/2, linear_data[method] + 4, f'+{improvement:.1f}%', 
                     ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.set_xlabel('SSL Methods', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Linear Classifier: SSL vs Frozen Supervised ViT', 
                  fontsize=16, fontweight='bold', pad=30)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, fontsize=12)
    ax1.set_ylim(0, 100)  # Increased y-limit for better spacing
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # Add value labels on bars - positioned to avoid overlap
    for i, method in enumerate(methods):
        # ViT baseline value - positioned in middle of bar
        ax1.text(i - width/2, linear_data['ViT']/2, f'{linear_data["ViT"]:.1f}%', 
                 ha='center', va='center', fontweight='bold', color='white', fontsize=10)
        # SSL method value - positioned in middle of bar
        ax1.text(i + width/2, linear_data[method]/2, f'{linear_data[method]:.1f}%', 
                 ha='center', va='center', fontweight='bold', color='white', fontsize=10)
    
    # Plot 2: Non-Linear Classifier Paired Comparisons
    for i, method in enumerate(methods):
        # ViT baseline bar
        ax2.bar(i - width/2, nonlinear_data['ViT'], width, 
                label='Frozen Supervised ViT' if i == 0 else "", 
                color=vit_color, alpha=0.8, edgecolor='black', linewidth=1)
        
        # SSL method bar
        ax2.bar(i + width/2, nonlinear_data[method], width, 
                label=f'{method} SSL' if i == 0 else "", 
                color=ssl_colors[i], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add improvement text above SSL bars - positioned higher to avoid overlap
        improvement = nonlinear_data[method] - nonlinear_data['ViT']
        if improvement > 0:
            ax2.text(i + width/2, nonlinear_data[method] + 4, f'+{improvement:.1f}%', 
                     ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax2.set_xlabel('SSL Methods', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Non-Linear Classifier: SSL vs Frozen Supervised ViT', 
                  fontsize=16, fontweight='bold', pad=30)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, fontsize=12)
    ax2.set_ylim(0, 100)  # Increased y-limit for better spacing
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # Add value labels on bars - positioned to avoid overlap
    for i, method in enumerate(methods):
        # ViT baseline value - positioned in middle of bar
        ax2.text(i - width/2, nonlinear_data['ViT']/2, f'{nonlinear_data["ViT"]:.1f}%', 
                 ha='center', va='center', fontweight='bold', color='white', fontsize=10)
        # SSL method value - positioned in middle of bar
        ax2.text(i + width/2, nonlinear_data[method]/2, f'{nonlinear_data[method]:.1f}%', 
                 ha='center', va='center', fontweight='bold', color='white', fontsize=10)
    
    # Add overall title with better positioning - moved higher up
    fig.suptitle('Self-Supervised Learning vs Frozen Supervised ViT:\nPaired Performance Comparison on CIFAR-10', 
                 fontsize=20, fontweight='bold', y=0.97)
    
    # Better spacing between subplots and overall layout - optimized without extra text
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.12, left=0.08, right=0.95, wspace=0.25)
    
    # Save the poster
    output_path = Path('results/final_evaluation/ssl_vs_vit_paired_comparison_poster.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ Clean paired comparison poster saved to: {output_path}")
    print("📊 Poster contains:")
    print("   • Left: Linear Classifier paired comparisons")
    print("   • Right: Non-Linear Classifier paired comparisons")
    print("   • Paired bars: ViT baseline vs each SSL method")
    print("   • Direct side-by-side comparison for each method")
    print("   • Improvement percentages above SSL bars")
    print("   • Professional styling for publications")
    print("   • Clean text layout with no overlapping")
    print("   • No extra text - clean and focused")
    
    plt.show()

def create_additional_visualizations():
    """Create additional comparison charts"""
    
    # Method comparison heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Data for heatmap
    methods = ['Frozen Supervised ViT', 'DINO SSL', 'MAE ViT (SSL)', 'iBOT SSL']
    categories = ['Linear Validation', 'Non-Linear Validation']
    
    data_matrix = np.array([
        [37.04, 36.92],  # ViT
        [79.38, 85.80],  # DINO
        [71.34, 86.66],  # MAE
        [68.73, 71.11]   # iBOT
    ])
    
    im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
    
    # Add text annotations with better positioning
    for i in range(len(methods)):
        for j in range(len(categories)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.1f}%', 
                          ha="center", va="center", color="black", fontweight='bold', fontsize=11)
    
    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(methods, fontsize=12)
    
    ax.set_title('SSL vs Frozen Supervised ViT: Validation Accuracy Matrix', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Validation Accuracy (%)', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    
    # Save heatmap
    output_path = Path('results/final_evaluation/ssl_vs_vit_validation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Clean validation accuracy heatmap saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    print("🚀 Creating Clean SSL vs Frozen Supervised ViT Paired Comparison Poster...")
    create_ssl_comparison_poster()
    
    print("\n🔍 Creating Additional Clean Visualizations...")
    create_additional_visualizations()
    
    print("\n🎉 All clean visualizations completed!")
    print("📁 Check results/final_evaluation/ for output files")
