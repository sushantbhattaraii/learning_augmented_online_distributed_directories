import pandas as pd
import matplotlib.pyplot as plt

# Filenames for the raw data
file_2 = "144nodes_diameter22_cutoff2.0-repetitions50-overlap100.xlsx"

# Load the datasets
df_2 = pd.read_excel(f"./../../results/grid_graphs/3/{file_2}")

# Group by fraction and calculate the mean for both metrics
summary_2 = df_2.groupby('fraction')[['stretch', 'stretch_arrow', 'stretch_parrow']].mean().reset_index()

# Combine data for plotting
cmap = plt.get_cmap('tab20')

# Create the plot
plt.figure(figsize=(2.35, 2.35*5/7), dpi=300)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

fractions = summary_2['fraction']

# Plotting the 4 lines
plt.plot([str(x) for x in fractions], summary_2['stretch_arrow'], marker='v', linestyle='-.', label=f'Arrow', color=cmap(2), linewidth=1.1, markersize=4, zorder=2)
plt.plot([str(x) for x in fractions], summary_2['stretch_parrow'], marker='^', linestyle=':', label=f'PArrow', color=cmap(4), linewidth=1.1, markersize=4, zorder=3)
plt.plot([str(x) for x in fractions], summary_2['stretch'], marker='.', linestyle='-', label=f'OPArrow', color=cmap(0), linewidth=1.1, markersize=4, zorder=1)


# Formatting the chart
plt.xlabel('Number of Operations', fontsize=9, labelpad=2)
plt.ylabel('Stretch', fontsize=9, labelpad=2)
# plt.title('Number of operations vs Mean Stretch for a network size n = 512', fontsize=12)
plt.legend(loc='upper left', 
           bbox_to_anchor=(0.0, 0.74),
           fontsize=8, frameon=True,
                borderpad=0.25,
                labelspacing=0.2,
                handletextpad=0.4,
                handlelength=2.2)
# plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks([str(x) for x in fractions])
plt.tight_layout(pad=0.05)

# Save and show
plt.savefig('third_grid.png')
# plt.show()

# Optional: Export summary to CSV
# combined.to_csv('summary_plot_fractions.csv', index=False)