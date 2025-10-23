import pandas as pd
import glob
import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.ticker import MultipleLocator
from fractions import Fraction
import os

# Gather and sort all Excel files

files = sorted(glob.glob('./results/threshEqualsDiam_data/random/1024nodes_diameter4_cutoff10*.xlsx'))

# Extract node count and overlap from filename
m = re.search(r'(\d+)nodes_', files[0])
node_count = m.group(1) if m else "?"

m2 = re.search(r'overlap(\d+)', files[0])
overlap_value = m2.group(1) if m2 else "?"

# Prepare colormap
cmap = plt.get_cmap('tab20')
all_x = []

# Create two subplots: one for error, one for stretch, one for stretch_arrow
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 11), sharex=True)

for idx, f in enumerate(files):
    df = pd.read_excel(f)
    mean_max_error = df.groupby('fraction')['max_error'].mean()
    # mean_min_error = df.groupby('fraction')['min_error'].mean()
    mean_stretch = df.groupby('fraction')['stretch'].mean()
    mean_stretch_arrow = df.groupby('fraction')['stretch_arrow'].mean()
    
    # mean_max_error.index = mean_max_error.index * int(node_count)
    # mean_min_error.index = mean_min_error.index * int(node_count)
    # mean_stretch.index = mean_stretch.index * int(node_count)
    # mean_stretch_arrow.index = mean_stretch_arrow.index * int(node_count)
    
    all_x.extend(mean_max_error.index.tolist())
    # all_x.extend(mean_min_error.index.tolist())
    
    # Extract cutoff value from filename
    cutoff = f.split("_cutoff")[1].split("-")[0]
    actual_cutoff = 1 / float(cutoff)

    # Plot error
    ax1.plot(
        mean_max_error.index,
        mean_max_error.values,
        marker='o',
        label=f'Prediction Error ≤ {actual_cutoff}',
        color=cmap(2 * idx)
    )

    # ax1.plot(
    #     mean_min_error.index,
    #     mean_min_error.values,
    #     marker='o',
    #     label=f'{actual_cutoff}-cutoff Error',
    #     color=cmap(2 * idx)
    # )
    # Plot stretch
    ax3.plot(
        mean_stretch.index,
        mean_stretch.values,
        marker='o',
        label=f'PArrow Stretch for Error ≤ {actual_cutoff}',
        color=cmap(2 * idx + 1)
    )

    ax3.plot(
        mean_stretch_arrow.index,
        mean_stretch_arrow.values,
        marker='o',
        label=f'Arrow Stretch for Error ≤ {actual_cutoff}',
        color=cmap(2 * idx + 2)
    )

    # ax4.plot(
    #     mean_stretch_arrow.index,
    #     mean_stretch_arrow.values,
    #     marker='o',
    #     label=f'Arrow Stretch for Error ≤ {actual_cutoff}',
    #     color=cmap(2 * idx + 2)
    # )

unique_x = sorted(set(all_x))

# Format max_error subplot
ax1.set_ylabel('Error')
# ax1.set_ylabel('Average of Max Error')
ax1.set_xlabel(f'Number of operations')
# ax1.set_xlabel(f'Number of predicted nodes among {node_count} nodes (# of operations)')
ax1.set_title(f'Error vs Number of operations')
# ax1.set_title(f'Error vs Fraction of Predicted Nodes')
ax1.legend(loc='best')
# ax1.grid(True)

# # Format min_error subplot
# ax2.set_ylabel('Average of Min Error')
# ax2.set_xlabel(f'Number of predicted nodes among {node_count} nodes (# of operations)')
# # ax1.set_title(f'Error vs Fraction of Predicted Nodes')
# ax2.legend(loc='best')
# # ax1.grid(True)

# Format stretch subplot
ax3.set_ylabel('PArrow and Arrow Stretch')
# ax3.set_xlabel(f'Number of predicted nodes among {node_count} nodes (# of operations)')
ax3.set_xlabel(f'Number of operations')
ax3.set_title(f'PArrow & Arrow Stretch vs Number of operations')
# ax3.set_title(f'PArrow & Arrow Stretch vs Fraction of Predicted Nodes')
ax3.legend(loc='best')
# ax2.grid(True)

# Format stretch_arrow subplot
# ax4.set_ylabel('Arrow Stretch')
# ax4.set_xlabel(f'Number of predicted nodes among {node_count} nodes (# of operations)')
# ax4.set_title(f'Arrow Stretch vs Fraction of Predicted Nodes')
# ax4.legend(loc='best')

# Set shared x-ticks
ax1.set_xticks(unique_x)
ax3.set_xticks(unique_x)
# ax4.set_xticks(unique_x)

# Save or display
plt.tight_layout()
plt.ylim(0, 5)
folder = "results/threshEqualsDiam_plots/random"

os.makedirs(folder, exist_ok=True)
# filename = f'{node_count}_nodes.png'
filename = f'{node_count}_nodes_cutoff_{actual_cutoff}.png'
path_to_save = os.path.join(folder, filename)

plt.savefig(path_to_save)
plt.show()
