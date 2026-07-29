import pandas as pd
import matplotlib.pyplot as plt

# File mapping for each node count (using the raw CSV versions of the uploaded files)
file_mapping = {
    128: "128nodes_diameter7_cutoff2.0-repetitions50-overlap100.xlsx",
    256: "256nodes_diameter5_cutoff2.0-repetitions50-overlap100.xlsx",
    512: "512nodes_diameter4_cutoff2.0-repetitions50-overlap100.xlsx",
    1024: "1024nodes_diameter4_cutoff2.0-repetitions50-overlap100.xlsx"
}

node_sizes = [128, 256, 512, 1024]

# Initialize lists to store mean values
stats = {
    # 'stretch_f128': [],
    'stretch_f512': [],
    # 'stretch_arrow_f128': [],
    'stretch_arrow_f512': [],
    'stretch_parrow_f512': []
}
cmap = plt.get_cmap('tab20')

# Process each file to compute means
for n in node_sizes:
    df = pd.read_excel(f"./../results/data_for_paper3/fourth/{file_mapping[n]}")
    
    # # Calculate means for fraction 128
    # stats['stretch_f128'].append(df[df['fraction'] == 128]['stretch'].mean())
    # stats['stretch_arrow_f128'].append(df[df['fraction'] == 128]['stretch_arrow'].mean())

    # Calculate means for fraction 512
    stats['stretch_f512'].append(df[df['fraction'] == 512]['stretch'].mean())
    stats['stretch_arrow_f512'].append(df[df['fraction'] == 512]['stretch_arrow'].mean())
    stats['stretch_parrow_f512'].append(df[df['fraction'] == 512]['stretch_parrow'].mean())

# Create the plot
plt.figure(figsize=(7, 5), dpi=180)
plt.xticks(fontsize=9)

# Plotting the 4 line graphs
plt.plot([str(x) for x in node_sizes], stats['stretch_arrow_f512'], marker='v', linestyle='-.', label=f'Arrow', color=cmap(2), linewidth=1)
plt.plot([str(x) for x in node_sizes], stats['stretch_parrow_f512'], marker='^', linestyle=':', label=f'PArrow', color=cmap(4), linewidth=1)
plt.plot([str(x) for x in node_sizes], stats['stretch_f512'], marker='.', linestyle='-', label=f'OPArrow', color=cmap(0), linewidth=1)
# plt.plot([str(x) for x in node_sizes], stats['stretch_f128'], marker='.', linestyle='-', label=f'Stretch$_{'O'}$$_{'P'}$$_{'A'}$$_{'r'}$$_{'r'}$$_{'o'}$$_{'w'} $(#${'opr'}$ = 128)', color=cmap(2), linewidth=1)
# plt.plot([str(x) for x in node_sizes], stats['stretch_arrow_f128'], marker='v', linestyle='-.', label=f'Stretch$_{'A'}$$_{'r'}$$_{'r'}$$_{'o'}$$_{'w'} $(#${'opr'}$ = 128)', color=cmap(3), linewidth=1)

# Formatting the plot
plt.xlabel('Network Size ($n$)', fontsize=12)
plt.ylabel('Stretch', fontsize=12)
# plt.title('Network size vs Mean Stretch for err $\leq$ 0.5', fontsize=12)
plt.legend(loc='lower right', fontsize=9)
# plt.grid(False, which='both', linestyle='--', alpha=0.7)
plt.xticks([str(x) for x in node_sizes])
plt.tight_layout()

# Save and display
# plt.savefig('stretch_metrics_plot.png')
plt.show()