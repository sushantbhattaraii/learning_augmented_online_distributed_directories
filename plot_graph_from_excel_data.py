import pandas as pd
import glob
import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.ticker import MultipleLocator
from fractions import Fraction
import os


# Gather and sort all 64-node Excel files
files = sorted(glob.glob('./results/error_and_stretch_data_with_cut-off_AND_overlap/1024/overlap50/1024nodes_diameter*.xlsx'))

m = re.search(r'(\d+)nodes_', files[0])
node_count = m.group(1) if m else "?"

m2 = re.search(r'(\d+).xlsx', files[0])
overlap_value = m2.group(1) if m else "?"

# Prepare colormap
cmap = plt.get_cmap('tab20')

all_x = []

plt.figure(figsize=(9, 10))
for idx, f in enumerate(files):
    # Read data
    df = pd.read_excel(f)
    # Compute mean error and mean stretch by fraction
    mean_error = df.groupby('fraction')['error'].mean()
    mean_stretch = df.groupby('fraction')['stretch'].mean()
    
    all_x.extend(mean_error.index.tolist())
    all_x.extend(mean_stretch.index.tolist())


    # Extract cutoff value from filename
    cutoff = f.split("_cutoff")[1].split("-")[0]
    actual_cutoff = 1/(float(cutoff))
    # Plot mean error with a unique color
    plt.plot(
        mean_error.index,
        mean_error.values,
        marker='o',
        label=f'{actual_cutoff}:cutoff Error',
        color=cmap(2 * idx)
    )
    # Plot mean stretch with the next unique color
    plt.plot(
        mean_stretch.index,
        mean_stretch.values,
        marker='o',
        label=f'{actual_cutoff}: cutoff Stretch',
        color=cmap(2 * idx + 1)
    )


unique_x = sorted(set(all_x))

# Labeling
plt.xlabel(f'Fraction of predicted nodes among {node_count} nodes (# of operations)')
plt.ylabel('Error / Stretch')
plt.title(f'Error/Stretch vs Fraction of nodes (# of operations) in {node_count} Node Graph | Prediction Overlap: {overlap_value}%')
# plt.grid(True)

ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.set_xticks(unique_x)


# y_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# for i in range(1, 41):
#     y_ticks.append(1 + i * 0.1)

# ax.set_yticks(y_ticks)
# ax.set_ylim(0.0, 4.0)

ymin = min([line.get_ydata().min() for line in ax.get_lines()])
ymax = max([line.get_ydata().max() for line in ax.get_lines()])
plt.ylim(ymin - 0.1, ymax + 0.1)  # small padding



plt.legend(loc='best')


folder = "6_cutoffs_merged"
filename = f'{node_count}_nodes_overlap{overlap_value}.png'
path_to_save = os.path.join('results', folder, filename)

# plt.savefig(path_to_save)
plt.show()
