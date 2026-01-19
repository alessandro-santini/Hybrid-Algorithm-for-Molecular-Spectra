#!/usr/bin/env python3
"""
Plot energy level comparison: exact (lines) vs projected (diamonds).
Generates: figures/all_molecules_peaks.pdf

Based on FancyPlot.ipynb
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import json
import string
import os

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
})

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "plot_data", "small_systems_plot_data.json")

with open(data_path, 'r') as f:
    data = json.load(f)

all_results = data['results']
letters = list(string.ascii_lowercase)

# Create 2x4 grid
nrows, ncols = 2, 4
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(3*ncols, 4*nrows),
                         constrained_layout=True)
axes = axes.flatten()

for i, (ax, res) in enumerate(zip(axes, all_results)):
    ax.annotate(f"({letters[i]})", xy=(0.875, 0.075), xycoords='axes fraction',
                fontsize=12, weight='bold', ha='left', va='top')

    y_full = np.array(res['omega_full'])
    y_proj = np.array(res['omega_proj'])

    # Exact lines
    for y in y_full:
        ax.hlines(y, 0, 1, color='black', lw=0.7)

    # Projected diamonds with jitter
    jitter = np.linspace(-0.03, 0.03, len(y_proj))
    for j, y in enumerate(y_proj):
        ax.plot(0.5 + jitter[j], y,
                marker='d', markersize=6,
                markeredgecolor='black',
                markerfacecolor='green')

    name = res['name'].removesuffix('1')
    title_plot = f"{name} ({res['base'].upper()}, R={res['R']} $\\AA$, $n_o$={res['no']}, $n_e$={res['ne']})"
    ax.set_title(title_plot, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xticks([])

    if i < 4:
        ax.set_ylim((0.0, 1.4))
    else:
        ax.set_ylim((0.0, 1.0))

    if i != 0 and i != 4:
        ax.set_yticks([])

# Turn off unused axes
for ax in axes[len(all_results):]:
    ax.axis('off')

# Y-labels
label_plot = r'$E-E_0\ (\rm{Ha})$'
axes[0].set_ylabel(label_plot, fontsize=14)
axes[4].set_ylabel(label_plot, fontsize=14)

# Legend
exact_line = mlines.Line2D([], [], color='black', lw=1, label='Exact')
approx_marker = mlines.Line2D([], [], color='green', marker='d', linestyle='None',
                              markersize=6, markeredgecolor='black', label='Projected')
axes[0].legend(handles=[exact_line, approx_marker], loc='lower left', fontsize=10, frameon=False)

# Save
fig.tight_layout()
output_dir = os.path.join(script_dir, "..", "figures")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "all_molecules_peaks.pdf")
fig.savefig(output_path)
print(f"Saved: {output_path}")

plt.show()
