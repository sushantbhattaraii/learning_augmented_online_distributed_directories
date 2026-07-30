import pandas as pd
import matplotlib.pyplot as plt

# Filenames for the raw data
file_5 = "2048nodes_diameter7_cutoff5.0-repetitions50-overlap100.xlsx"

# Load the datasets
df_5 = pd.read_excel(f"./../../results/internet_graphs/3/{file_5}")

# Group by fraction and calculate the mean for both metrics
summary_5 = df_5.groupby('fraction')[['stretch', 'stretch_arrow', 'stretch_parrow']].mean().reset_index()

# Combine data for plotting
cmap = plt.get_cmap('tab20')

# Create the plot
plt.figure(figsize=(2.35, 2.35*5/7), dpi=300)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

fractions = summary_5['fraction']

# Plotting the 4 lines
plt.plot([str(x) for x in fractions], summary_5['stretch_arrow'], marker='v', linestyle='-.', label=f'Arrow', color=cmap(2), linewidth=1.1, markersize=4, zorder=2)
plt.plot([str(x) for x in fractions], summary_5['stretch_parrow'], marker='^', linestyle=':', label=f'PArrow', color=cmap(4), linewidth=1.1, markersize=4, zorder=3)
plt.plot([str(x) for x in fractions], summary_5['stretch'], marker='.', linestyle='-', label=f'OPArrow', color=cmap(0), linewidth=1.1, markersize=4, zorder=1)


# Formatting the chart
plt.xlabel('Number of Operations', fontsize=9, labelpad=2)
plt.ylabel('Stretch', fontsize=9, labelpad=2)
# plt.title('Number of operations vs Mean Stretch for a network size n = 512', fontsize=12)
plt.legend(loc='upper left', 
           bbox_to_anchor=(0.0, 0.9),
           fontsize=8, frameon=True,
                borderpad=0.25,
                labelspacing=0.2,
                handletextpad=0.4,
                handlelength=2.2)
# plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks([str(x) for x in fractions])
plt.tight_layout(pad=0.05)

# Save and show
# plt.savefig('third_internet.png')
plt.show()

# Optional: Export summary to CSV
# combined.to_csv('summary_plot_fractions.csv', index=False)