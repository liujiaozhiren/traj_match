import matplotlib.pyplot as plt
import numpy as np

# Prepare data
np.random.seed(0)
num_points = 50
data = [np.cumsum(np.random.randn(num_points)) for _ in range(5)]
labels = [' ', ' ', ' ', ' ', ' ']

# Create figure with 5 tight subplots (one per row)
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 8), sharex=True)
plt.subplots_adjust(hspace=0.05)  # minimize vertical space

for ax, y, lbl in zip(axes, data, labels):
    ax.plot(y, linewidth=2)

    # Add label in the top‑left corner
    ax.text(0.02, 0.90, lbl, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')

    # Remove grids, ticks, and tick labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Ensure all spines (outer frame) are visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

plt.show()