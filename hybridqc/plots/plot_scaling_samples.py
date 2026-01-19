#!/usr/bin/env python3
"""
Plot scaling of excitation energy errors with number of samples.
Generates: figures/scaling_samples.pdf

Based on ScalingSamples.ipynb
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
})

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "plot_data", "scaling_samples_peak_errors.json")

with open(data_path, 'r') as f:
    data = json.load(f)

peak_errors = data['peak_errors']
molecules_info = data['molecules']
sample_exponents = data['sample_exponents']
n_peaks = data['n_peaks']
df = data['resolution_limit']

# Color and marker settings
cpalette = sns.diverging_palette(250, 30, l=65, center="dark", n=4)
markers = ['o', 'v', '*', '^']
label_lgd = ["1st excited state", "2nd excited state", "3rd excited state", "4th excited state"]

# Create 1x2 plot (HCl and LiH only)
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, mol_name in zip(axs, ["HCl", "LiH1"]):
    if mol_name not in peak_errors:
        continue

    info = molecules_info[mol_name]
    no = info['no']
    ne = info['ne']
    R = info['R']
    base = info['base']

    # Calculate mean and std for each sample size
    error_peak_means = []
    error_peak_std = []

    for i in sample_exponents:
        n_samples = str(2 ** i)
        if n_samples in peak_errors[mol_name]:
            errors = np.array(peak_errors[mol_name][n_samples])
            error_peak_means.append(np.nanmean(errors, axis=0))
            error_peak_std.append(np.nanstd(errors, axis=0))
        else:
            error_peak_means.append([np.nan] * n_peaks)
            error_peak_std.append([np.nan] * n_peaks)

    error_peak_means = np.array(error_peak_means)
    error_peak_std = np.array(error_peak_std)

    # Plot
    x_vals = [2**i for i in sample_exponents]
    for i in range(n_peaks):
        ax.errorbar(x_vals,
                    error_peak_means[:, i],
                    yerr=error_peak_std[:, i] / np.sqrt(15),
                    linestyle='-', marker=markers[i], ms=8, linewidth=1, capsize=3,
                    label=label_lgd[i], color=cpalette[i], markerfacecolor="none")

    ax.legend(fontsize=11)
    ax.hlines(df, 2**sample_exponents[0], 2**sample_exponents[-1],
              color='gray', linestyle='--')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    title = f"{mol_name.removesuffix('1')} ({base.upper()}, R={R} $\\AA$, $n_o={no}$, $n_e={ne}$)"
    ax.set_title(title, fontsize=13)
    ax.grid(True)
    ax.set_xticks([2**i for i in sample_exponents])

# Axis labels
for ax in axs:
    ax.set_ylabel("Error Excited states", fontsize=13)
    ax.set_xlabel("Number of Samples", fontsize=13)

# Save
plt.tight_layout()
output_dir = os.path.join(script_dir, "..", "figures")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "scaling_samples.pdf")
fig.savefig(output_path, bbox_inches='tight')
print(f"Saved: {output_path}")

plt.show()
