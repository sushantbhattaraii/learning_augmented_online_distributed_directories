import pandas as pd
import matplotlib.pyplot as plt

# Define categories and the corresponding cutoff strings found in filenames
categories = ['$0.0$', '$0.1$', '$0.2$', '$0.3$', '$0.4$', '$0.5$']
cutoffs = ['inf', '10.0', '5.0', '3.3333333333333335', '2.5', '2.0']

# Data structures to store results for 4 line graphs
results = {
    'Category': categories,
    # 'mean_stretch_256': [],
    # 'mean_stretch_arrow_256': [],
    # 'mean_stretch_parrow_256': [],
    # 'mean_stretch_1024': [],
    # 'mean_stretch_arrow_1024': [],
    # 'mean_stretch_parrow_1024': [],
    'mean_stretch_512': [],
    'mean_stretch_arrow_512': [],
    'mean_stretch_parrow_512': [],
}
cmap = plt.get_cmap('tab20')

# Process files for each category
for cutoff in cutoffs:
    # Construct filenames for 256 and 1024 nodes
    # file_256 = f"256nodes_diameter41_cutoff{cutoff}-repetitions50-overlap100.xlsx"
    file_512 = f"512nodes_diameter106_cutoff{cutoff}-repetitions50-overlap100.xlsx"
    # file_1024 = f"1024nodes_diameter7_cutoff{cutoff}-repetitions50-overlap100.xlsx"
    
    # Load and aggregate data for 256 nodes
    # # df_256 = pd.read_excel(file_256)
    # df_256 = pd.read_excel(f"./../results/data_for_paper3/second/{file_256}")
    # results['mean_stretch_256'].append(df_256['stretch'].mean())
    # results['mean_stretch_arrow_256'].append(df_256['stretch_arrow'].mean())
    # results['mean_stretch_parrow_256'].append(df_256['stretch_parrow'].mean())
    
    # Load and aggregate data for 512 nodes
    df_512 = pd.read_excel(f"./../../results/small_world_graphs/first_way/2/{file_512}")
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

# plt.plot([str(x) for x in categories], results['mean_stretch_arrow_256'], marker='v', linestyle='-.', label=f'Arrow', color=cmap(2), linewidth=1)
# plt.plot([str(x) for x in categories], results['mean_stretch_parrow_256'], marker='^', linestyle=':', label=f'PArrow', color=cmap(4), linewidth=1)
# plt.plot([str(x) for x in categories], results['mean_stretch_256'], marker='.', linestyle='-', label=f'OPArrow', color=cmap(0), linewidth=1)

# plt.plot([str(x) for x in categories], results['mean_stretch_256'], marker='.', linestyle='-', label=f'Stretch$_{'O'}$$_{'P'}$$_{'A'}$$_{'r'}$$_{'r'}$$_{'o'}$$_{'w'}$ (n = 256)', color=cmap(2), linewidth=1)
# plt.plot([str(x) for x in categories], results['mean_stretch_arrow_256'], marker='v', linestyle='-.', label=f'Stretch$_{'A'}$$_{'r'}$$_{'r'}$$_{'o'}$$_{'w'} $(n = 256)', color=cmap(3), linewidth=1)


# Labels and Formatting
plt.xlabel('Error', fontsize=9, labelpad=2)
plt.ylabel('Stretch', fontsize=9, labelpad=2)
# plt.title('Maximum Error Bound vs Mean Stretch for $\# opr = 256$', fontsize=12)

plt.legend(loc='upper right', 
                bbox_to_anchor=(1.0, 0.85), 
                fontsize=8, frameon=True,
                borderpad=0.25,
                labelspacing=0.2,
                handletextpad=0.4,
                handlelength=2.2)


# plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks([str(x) for x in categories])
plt.tight_layout(pad=0.05)

# Save the plot
# plt.savefig('stretch_metrics_by_error_bound.png')
plt.show()

# Output the summary to CSV
# pd.DataFrame(results).to_csv('summary_error_bounds.csv', index=False)