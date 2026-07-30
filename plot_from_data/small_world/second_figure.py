import pandas as pd
import matplotlib.pyplot as plt

# Define categories and the corresponding cutoff strings found in filenames
categories = ['$0.0$', '$0.1$', '$0.2$', '$0.3$', '$0.4$', '$0.5$']
cutoffs = ['inf', '10.0', '5.0', '3.3333333333333335', '2.5', '2.0']

# Data structures to store results for 4 line graphs
results = {
    'Category': categories,
    'mean_stretch_512': [],
    'mean_stretch_arrow_512': [],
    'mean_stretch_parrow_512': [],
}
cmap = plt.get_cmap('tab20')

# Process files for each category
for cutoff in cutoffs:
    # Construct filenames for 512 nodes
    file_512 = f"512nodes_diameter106_cutoff{cutoff}-repetitions50-overlap100.xlsx"
    
    # Load and aggregate data for 512 nodes
    df_512 = pd.read_excel(f"./../../results/small_world_graphs/2/{file_512}")
    results['mean_stretch_512'].append(df_512['stretch'].mean())
    results['mean_stretch_arrow_512'].append(df_512['stretch_arrow'].mean())
    results['mean_stretch_parrow_512'].append(df_512['stretch_parrow'].mean())

# Create the plot
plt.figure(figsize=(2.35, 2.35*5/7), dpi=300)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Plotting the 4 lines
plt.plot([str(x) for x in categories], results['mean_stretch_arrow_512'], marker='v', linestyle='-.', label=f'Arrow', color=cmap(2), linewidth=1.1, markersize=4, zorder=2)
plt.plot([str(x) for x in categories], results['mean_stretch_parrow_512'], marker='^', linestyle=':', label=f'PArrow', color=cmap(4), linewidth=1.1, markersize=4, zorder=3)
plt.plot([str(x) for x in categories], results['mean_stretch_512'], marker='.', linestyle='-', label=f'OPArrow', color=cmap(0), linewidth=1.1, markersize=4, zorder=1)

# Labels and Formatting
plt.xlabel('Error($\delta$)', fontsize=9, labelpad=2)
plt.ylabel('Stretch', fontsize=9, labelpad=2)
# plt.title('Maximum Error Bound vs Mean Stretch for $\# opr = 256$', fontsize=12)

plt.legend(loc='upper right', 
                bbox_to_anchor=(1.0, 0.8), 
                fontsize=8, frameon=True,
                borderpad=0.25,
                labelspacing=0.2,
                handletextpad=0.4,
                handlelength=2.2)


# plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks([str(x) for x in categories])
plt.tight_layout(pad=0.05)

# Save the plot
plt.savefig('second_small_world.png')
# plt.show()

# Output the summary to CSV
# pd.DataFrame(results).to_csv('summary_error_bounds.csv', index=False)