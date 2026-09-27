#!/usr/bin/env python3
"""
Generate plots for the light speed deviation paper.
All plots are exported both as PNG and PDF.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm, lognorm, weibull_min, gamma
from scipy.integrate import quad
import json

matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (10, 6)

# Color scheme
COLOR_PRIMARY = '#1E4C8C'
COLOR_SECONDARY = '#B43232'
COLOR_ACCENT = '#2A7838'
COLOR_LIGHT = '#F5F5F8'

def ensure_output_dir():
    import os
    os.makedirs('/home/claude/plots', exist_ok=True)

# ============================================================================
# Plot 1: Refractive Index vs. Wavelength in Different Media
# ============================================================================
def plot1_refractive_index():
    """Plot 1: Refractive indices in different media."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    wavelength = np.linspace(300, 1600, 500)  # nm
    
    # Physical models for refractive indices (Sellmeier-like)
    n_glass = 1.5 + 0.03 * np.exp(-((wavelength - 500)/200)**2)
    n_water = 1.33 + 0.02 * np.exp(-((wavelength - 450)/180)**2)
    n_diamond = 2.4 + 0.05 * np.exp(-((wavelength - 550)/150)**2)
    n_air = 1.0003 + 0.0001 * np.exp(-((wavelength - 600)/300)**2)
    
    ax.plot(wavelength, n_glass, label='Glass', linewidth=2.5, 
            color=COLOR_PRIMARY, marker='', markersize=0)
    ax.plot(wavelength, n_water, label='Water', linewidth=2.5, 
            color=COLOR_SECONDARY, marker='', markersize=0)
    ax.plot(wavelength, n_diamond, label='Diamond', linewidth=2.5, 
            color=COLOR_ACCENT, marker='', markersize=0)
    ax.plot(wavelength, n_air, label='Air', linewidth=2.5, 
            color='gray', linestyle='--', marker='', markersize=0)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Refractive Index n', fontsize=12, fontweight='bold')
    ax.set_title('Refractive Index Variation Across Wavelengths', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.set_facecolor(COLOR_LIGHT)
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig('/home/claude/plots/plot1_refractive_index.png', dpi=150, bbox_inches='tight')
    plt.savefig('/home/claude/plots/plot1_refractive_index.pdf', bbox_inches='tight')
    plt.close()

# ============================================================================
# Plot 2: Light Speed Deviation Distribution
# ============================================================================
def plot2_speed_deviation():
    """Plot 2: Distribution of light speed deviations in different media."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    c = 3e8  # m/s
    
    # Scenario 1: Glass - Normal distribution centered on c/n
    ax = axes[0, 0]
    v_glass = np.linspace(1.8e8, 2.2e8, 1000)
    mu_glass = c / 1.5
    sigma_glass = 0.08e8
    pdf_glass = norm.pdf(v_glass, mu_glass, sigma_glass)
    ax.fill_between(v_glass/1e8, pdf_glass*1e-8, alpha=0.5, color=COLOR_PRIMARY)
    ax.plot(v_glass/1e8, pdf_glass*1e-8, linewidth=2.5, color=COLOR_PRIMARY)
    ax.axvline(mu_glass/1e8, color=COLOR_SECONDARY, linestyle='--', linewidth=2, label='Mean')
    ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax.set_title('Glass (n=1.5)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor(COLOR_LIGHT)
    ax.legend()
    
    # Scenario 2: Water - Slightly different distribution
    ax = axes[0, 1]
    v_water = np.linspace(2.2e8, 2.35e8, 1000)
    mu_water = c / 1.33
    sigma_water = 0.06e8
    pdf_water = norm.pdf(v_water, mu_water, sigma_water)
    ax.fill_between(v_water/1e8, pdf_water*1e-8, alpha=0.5, color=COLOR_SECONDARY)
    ax.plot(v_water/1e8, pdf_water*1e-8, linewidth=2.5, color=COLOR_SECONDARY)
    ax.axvline(mu_water/1e8, color=COLOR_PRIMARY, linestyle='--', linewidth=2, label='Mean')
    ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax.set_title('Water (n=1.33)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor(COLOR_LIGHT)
    ax.legend()
    
    # Scenario 3: Diamond - Log-normal distribution (dispersion)
    ax = axes[1, 0]
    v_diamond = np.linspace(1.1e8, 1.35e8, 1000)
    s_param = 0.25
    scale_param = 1.25e8
    pdf_diamond = lognorm.pdf(v_diamond, s_param, scale=scale_param)
    ax.fill_between(v_diamond/1e8, pdf_diamond*1e-8, alpha=0.5, color=COLOR_ACCENT)
    ax.plot(v_diamond/1e8, pdf_diamond*1e-8, linewidth=2.5, color=COLOR_ACCENT)
    ax.set_xlabel('Light Speed (10⁸ m/s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax.set_title('Diamond (n=2.4, Log-normal)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor(COLOR_LIGHT)
    
    # Scenario 4: Comparative deviations
    ax = axes[1, 1]
    media = ['Glass', 'Water', 'Diamond', 'Air']
    refractive_indices = [1.5, 1.33, 2.4, 1.0003]
    speeds = [c/n/1e8 for n in refractive_indices]
    deviations = [(c/1e8 - v) for v in speeds]
    
    bars = ax.bar(media, deviations, color=[COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, 'gray'], 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Deviation from c (10⁸ m/s)', fontsize=11, fontweight='bold')
    ax.set_title('Speed Deviations by Medium', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_facecolor(COLOR_LIGHT)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    fig.suptitle('Light Speed Deviation Distribution Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig('/home/claude/plots/plot2_speed_deviation.png', dpi=150, bbox_inches='tight')
    plt.savefig('/home/claude/plots/plot2_speed_deviation.pdf', bbox_inches='tight')
    plt.close()

# ============================================================================
# Plot 3: Maximum and Expected Deviations
# ============================================================================
def plot3_deviation_analysis():
    """Plot 3: Maximum and expected deviations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    c = 3e8
    
    # Refractive indices for various media
    media_data = {
        'Air': (1.0003, 0.00001),
        'Water': (1.33, 0.02),
        'Glass': (1.5, 0.03),
        'Fused Silica': (1.456, 0.025),
        'Diamond': (2.4, 0.05),
        'Dense Flint': (1.8, 0.04),
    }
    
    # Calculate deviations
    max_deviations = []
    expected_deviations = []
    media_names = []
    
    for medium, (n_mean, n_std) in media_data.items():
        media_names.append(medium)
        v_mean = c / n_mean
        # Maximum deviation: c - c/n_min (when n is at its minimum)
        max_dev = (c - c/(n_mean + 3*n_std))/1e8
        # Expected deviation: c - c/n_mean
        exp_dev = (c - v_mean)/1e8
        max_deviations.append(max_dev)
        expected_deviations.append(exp_dev)
    
    # Plot 1: Comparison
    x_pos = np.arange(len(media_names))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, max_deviations, width, label='Maximum Deviation',
                    color=COLOR_PRIMARY, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x_pos + width/2, expected_deviations, width, label='Expected Deviation',
                    color=COLOR_SECONDARY, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Medium', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Deviation (10⁸ m/s)', fontsize=12, fontweight='bold')
    ax1.set_title('Maximum vs Expected Deviations', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(media_names, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_facecolor(COLOR_LIGHT)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Deviation as percentage
    percentages = [(exp/expected_deviations[-1])*100 for exp in expected_deviations]
    
    bars = ax2.barh(media_names, percentages, color=COLOR_ACCENT, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Relative Deviation (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Relative Deviations (normalized to Diamond)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax2.set_facecolor(COLOR_LIGHT)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        ax2.text(pct, bar.get_y() + bar.get_height()/2, f'{pct:.1f}%',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    fig.suptitle('Deviation Analysis Across Media', fontsize=16, fontweight='bold', y=1.00)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig('/home/claude/plots/plot3_deviation_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig('/home/claude/plots/plot3_deviation_analysis.pdf', bbox_inches='tight')
    plt.close()

# ============================================================================
# Plot 4: Cumulative Distribution Functions
# ============================================================================
def plot4_cdf():
    """Plot 4: Cumulative distribution functions for different media."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    c = 3e8
    
    # CDF for Glass
    ax = axes[0, 0]
    v_glass = np.linspace(1.8e8, 2.2e8, 1000)
    mu_glass = c / 1.5
    sigma_glass = 0.08e8
    cdf_glass = norm.cdf(v_glass, mu_glass, sigma_glass)
    ax.plot(v_glass/1e8, cdf_glass, linewidth=2.5, color=COLOR_PRIMARY)
    ax.fill_between(v_glass/1e8, 0, cdf_glass, alpha=0.3, color=COLOR_PRIMARY)
    ax.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax.set_title('Glass CDF', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor(COLOR_LIGHT)
    ax.set_ylim([0, 1])
    
    # CDF for Water
    ax = axes[0, 1]
    v_water = np.linspace(2.2e8, 2.35e8, 1000)
    mu_water = c / 1.33
    sigma_water = 0.06e8
    cdf_water = norm.cdf(v_water, mu_water, sigma_water)
    ax.plot(v_water/1e8, cdf_water, linewidth=2.5, color=COLOR_SECONDARY)
    ax.fill_between(v_water/1e8, 0, cdf_water, alpha=0.3, color=COLOR_SECONDARY)
    ax.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax.set_title('Water CDF', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor(COLOR_LIGHT)
    ax.set_ylim([0, 1])
    
    # CDF for Diamond
    ax = axes[1, 0]
    v_diamond = np.linspace(1.1e8, 1.35e8, 1000)
    s_param = 0.25
    scale_param = 1.25e8
    cdf_diamond = lognorm.cdf(v_diamond, s_param, scale=scale_param)
    ax.plot(v_diamond/1e8, cdf_diamond, linewidth=2.5, color=COLOR_ACCENT)
    ax.fill_between(v_diamond/1e8, 0, cdf_diamond, alpha=0.3, color=COLOR_ACCENT)
    ax.set_xlabel('Light Speed (10⁸ m/s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax.set_title('Diamond CDF', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor(COLOR_LIGHT)
    ax.set_ylim([0, 1])
    
    # Comparative CDF
    ax = axes[1, 1]
    v_range = np.linspace(1.0e8, 3.0e8, 500)
    # Normalize to 0-1 range for comparison
    cdf_glass_comp = norm.cdf(v_range, c/1.5, 0.08e8)
    cdf_water_comp = norm.cdf(v_range, c/1.33, 0.06e8)
    cdf_diamond_comp = lognorm.cdf(v_range, 0.25, scale=1.25e8)
    
    ax.plot(v_range/1e8, cdf_glass_comp, linewidth=2.5, color=COLOR_PRIMARY, label='Glass')
    ax.plot(v_range/1e8, cdf_water_comp, linewidth=2.5, color=COLOR_SECONDARY, label='Water')
    ax.plot(v_range/1e8, cdf_diamond_comp, linewidth=2.5, color=COLOR_ACCENT, label='Diamond')
    ax.set_xlabel('Light Speed (10⁸ m/s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax.set_title('Comparative CDFs', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor(COLOR_LIGHT)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    
    fig.suptitle('Cumulative Distribution Functions', fontsize=16, fontweight='bold', y=0.995)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig('/home/claude/plots/plot4_cdf.png', dpi=150, bbox_inches='tight')
    plt.savefig('/home/claude/plots/plot4_cdf.pdf', bbox_inches='tight')
    plt.close()

# ============================================================================
# Plot 5: Dispersion Effects
# ============================================================================
def plot5_dispersion():
    """Plot 5: Dispersion (wavelength-dependent speed variation)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    wavelength = np.linspace(300, 1000, 200)  # nm
    c = 3e8
    
    # Cauchy's equation for glass: n(λ) = A + B/λ²
    A_glass = 1.5
    B_glass = 0.003
    n_glass = A_glass + B_glass / (wavelength/1000)**2
    v_glass = c / n_glass
    
    # Water dispersion
    A_water = 1.33
    B_water = 0.002
    n_water = A_water + B_water / (wavelength/1000)**2
    v_water = c / n_water
    
    # Deviation from mean
    dev_glass = v_glass - np.mean(v_glass)
    dev_water = v_water - np.mean(v_water)
    
    # Plot 1: Speed vs Wavelength
    ax1.plot(wavelength, v_glass/1e8, linewidth=2.5, color=COLOR_PRIMARY, 
             label='Glass', marker='o', markersize=3, markevery=15)
    ax1.plot(wavelength, v_water/1e8, linewidth=2.5, color=COLOR_SECONDARY, 
             label='Water', marker='s', markersize=3, markevery=15)
    ax1.fill_between(wavelength, v_glass/1e8, np.mean(v_glass)/1e8, alpha=0.2, color=COLOR_PRIMARY)
    ax1.fill_between(wavelength, v_water/1e8, np.mean(v_water)/1e8, alpha=0.2, color=COLOR_SECONDARY)
    ax1.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Light Speed (10⁸ m/s)', fontsize=12, fontweight='bold')
    ax1.set_title('Dispersion: Speed vs Wavelength', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor(COLOR_LIGHT)
    ax1.legend(fontsize=11)
    
    # Plot 2: Deviation from mean
    ax2.plot(wavelength, dev_glass/1e6, linewidth=2.5, color=COLOR_PRIMARY, 
             label='Glass', marker='o', markersize=3, markevery=15)
    ax2.plot(wavelength, dev_water/1e6, linewidth=2.5, color=COLOR_SECONDARY, 
             label='Water', marker='s', markersize=3, markevery=15)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.fill_between(wavelength, dev_glass/1e6, 0, alpha=0.2, color=COLOR_PRIMARY, where=(dev_glass>=0))
    ax2.fill_between(wavelength, dev_glass/1e6, 0, alpha=0.2, color=COLOR_PRIMARY, where=(dev_glass<0))
    ax2.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Deviation from Mean (10⁶ m/s)', fontsize=12, fontweight='bold')
    ax2.set_title('Dispersion: Deviation from Mean Speed', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor(COLOR_LIGHT)
    ax2.legend(fontsize=11)
    
    fig.suptitle('Dispersion Effects in Different Media', fontsize=16, fontweight='bold', y=1.00)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig('/home/claude/plots/plot5_dispersion.png', dpi=150, bbox_inches='tight')
    plt.savefig('/home/claude/plots/plot5_dispersion.pdf', bbox_inches='tight')
    plt.close()

# ============================================================================
# Plot 6: Statistical Summary
# ============================================================================
def plot6_statistics():
    """Plot 6: Statistical summary of deviations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    c = 3e8
    media_data = {
        'Air': (1.0003, 0.00001),
        'Water': (1.33, 0.02),
        'Glass': (1.5, 0.03),
        'Diamond': (2.4, 0.05),
    }
    
    # Collect statistics
    means = []
    stds = []
    mins = []
    maxs = []
    media_names = list(media_data.keys())
    
    for medium, (n_mean, n_std) in media_data.items():
        v_mean = c / n_mean
        # Assume normal distribution
        samples = np.random.normal(v_mean, n_std*1e7, 10000)
        means.append(v_mean/1e8)
        stds.append(np.std(samples)/1e8)
        mins.append(np.min(samples)/1e8)
        maxs.append(np.max(samples)/1e8)
    
    # Plot 1: Box plot
    ax = axes[0, 0]
    box_data = []
    for medium, (n_mean, n_std) in media_data.items():
        v_mean = c / n_mean
        samples = np.random.normal(v_mean, n_std*1e7, 1000)
        box_data.append(samples/1e8)
    
    bp = ax.boxplot(box_data, labels=media_names, patch_artist=True,
                    medianprops=dict(color='red', linewidth=2),
                    boxprops=dict(facecolor=COLOR_PRIMARY, alpha=0.7),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    ax.set_ylabel('Light Speed (10⁸ m/s)', fontsize=11, fontweight='bold')
    ax.set_title('Box Plot of Speed Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_facecolor(COLOR_LIGHT)
    
    # Plot 2: Mean and Standard Deviation
    ax = axes[0, 1]
    x_pos = np.arange(len(media_names))
    ax.errorbar(x_pos, means, yerr=stds, fmt='o', markersize=8, capsize=5,
               capthick=2, linewidth=2.5, color=COLOR_PRIMARY, 
               ecolor=COLOR_SECONDARY, label='Mean ± Std')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(media_names)
    ax.set_ylabel('Light Speed (10⁸ m/s)', fontsize=11, fontweight='bold')
    ax.set_title('Mean and Standard Deviation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_facecolor(COLOR_LIGHT)
    ax.legend(fontsize=10)
    
    # Plot 3: Standard Deviation Comparison
    ax = axes[1, 0]
    bars = ax.bar(media_names, stds, color=COLOR_ACCENT, alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Standard Deviation (10⁸ m/s)', fontsize=11, fontweight='bold')
    ax.set_title('Variability Across Media', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_facecolor(COLOR_LIGHT)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Coefficient of Variation
    ax = axes[1, 1]
    cv = [(stds[i]/means[i])*100 for i in range(len(media_names))]
    bars = ax.barh(media_names, cv, color=COLOR_PRIMARY, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Coefficient of Variation (%)', fontsize=11, fontweight='bold')
    ax.set_title('Relative Variability (CV)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.set_facecolor(COLOR_LIGHT)
    
    for i, (bar, val) in enumerate(zip(bars, cv)):
        ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}%',
               ha='left', va='center', fontsize=10, fontweight='bold')
    
    fig.suptitle('Statistical Summary of Light Speed Variations', 
                fontsize=16, fontweight='bold', y=0.995)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig('/home/claude/plots/plot6_statistics.png', dpi=150, bbox_inches='tight')
    plt.savefig('/home/claude/plots/plot6_statistics.pdf', bbox_inches='tight')
    plt.close()

# ============================================================================
# Generate all plots
# ============================================================================
if __name__ == '__main__':
    ensure_output_dir()
    print("Generating Plot 1: Refractive Index...")
    plot1_refractive_index()
    print("Generating Plot 2: Speed Deviation Distribution...")
    plot2_speed_deviation()
    print("Generating Plot 3: Deviation Analysis...")
    plot3_deviation_analysis()
    print("Generating Plot 4: CDF...")
    plot4_cdf()
    print("Generating Plot 5: Dispersion...")
    plot5_dispersion()
    print("Generating Plot 6: Statistics...")
    plot6_statistics()
    print("All plots generated successfully!")
    print("Plots available in: /home/claude/plots/")
