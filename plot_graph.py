import networkx as nx
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import math
import pandas as pd

def show_graph(G):
    # Load the graph from a GraphML file
    # graphml_file = '.\\graphs\\10random_diameter2test.edgelist'
    # G = nx.read_graphml(graphml_file)

    # Position nodes for better visualization
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # # Draw node labels
    nx.draw_networkx_labels(G, pos)

    # Show the plot
    plt.show()


def plot_error_graph(fraction, errors_to_plot, file_name):

    x = fraction
    y = errors_to_plot

    numbers_in_filename = re.findall(r'\d+\.?\d*', file_name)
    total_number_of_nodes_in_graph = str(numbers_in_filename[0])

    x = [element * int(total_number_of_nodes_in_graph) for element in x]

    # print(type(x), type(y))
    # exit(0)

    plt.figure()
    plt.plot(x, y, marker='o', linestyle='-')

    # Draw vertical lines from each point to the x-axis and add labels
    for xi, yi in zip(x, y):
        plt.vlines(x=xi, ymin=0, ymax=yi, color='gray', linestyle='--', linewidth=0.8)
        plt.text(int(xi), -0.02 * max(y), f'{xi}', ha='center', va='top', fontsize=8, rotation=90)

    plt.xlabel('Number of Predicted Nodes/#of operations')
    plt.ylabel('Error')
    plt.title('# of operations vs Error Graph | Number of nodes: '+ total_number_of_nodes_in_graph)
    plt.grid(True)
    folder_name = "results"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    path_to_save = os.path.join('results', 'error_graphs', file_name)
    plt.savefig(path_to_save)
    plt.show()


def plot_error_and_stretch_graph_with_boxplot(fractions, errors, file_name, reps, stretches, error_cutoff, overlap):
    n_groups = len(fractions)

    # 1) Group by taking every 5th element, starting at offsets 0–4
    groups = [errors[i::n_groups] for i in range(n_groups)]
    print("Printing groups of errors:")
    print(groups)

    groups2 = [stretches[i::n_groups] for i in range(n_groups)]
    print("Printing groups of stretches:")
    print(groups2)

    # 2) Compute mean of each group
    means = np.array([sum(g)/len(g) for g in groups])
    maxes = np.array([max(g) for g in groups])
    mins = np.array([min(g) for g in groups])

    means_stretch = np.array([sum(g)/len(g) for g in groups2])
    maxes_stretch = np.array([max(g) for g in groups2])
    mins_stretch = np.array([min(g) for g in groups2])

    print("Printing maxes and mins of errors:")
    print(maxes, mins)
    print("Printing maxes and mins of stretches:")
    print(maxes_stretch, mins_stretch)

    x = fractions

    numbers_in_filename = re.findall(r'\d+\.?\d*', file_name)
    total_number_of_nodes_in_graph = str(numbers_in_filename[0])

    x = [element * int(total_number_of_nodes_in_graph) for element in x]

    yerr  = np.vstack([means - mins, maxes - means])
    yerr2  = np.vstack([means_stretch - mins_stretch, maxes_stretch - means_stretch])
    plt.figure(figsize=(14, 10))
    # plt.plot(x, mean_y, marker='o', linestyle='-')
    # plt.plot(x, max_y, marker='o', linestyle='-')
    # plt.plot(x, min_y, marker='o', linestyle='-')

    plt.errorbar(
    x, means,
    yerr=yerr,
    fmt='-s',          # line + square marker
    color='red',        # pick a single color
    ecolor='red',       # same color for errorbars
    capsize=5,         # width of the caps on the whiskers
    markersize=8,
    lw=2,
    label='Error'
    )

    plt.errorbar(
    x, means_stretch,
    yerr=yerr2,
    fmt='-s',          # line + square marker
    color='green',        # pick a single color
    ecolor='green',       # same color for errorbars
    capsize=5,         # width of the caps on the whiskers
    markersize=8,
    lw=2,
    label='Stretch'

    )

    # Draw vertical lines from each point to the x-axis and add labels
    for xi, yi in zip(x, means):
        yi = round(yi, 4)
        plt.vlines(x=xi, ymin=0, ymax=yi, color='gray', linestyle='--', linewidth=0.8)
        plt.text(int(xi), -0.02 * max(means), f'{xi}', ha='center', va='top', fontsize=8, rotation=90)
        plt.text(xi, yi, str(yi), ha='left', va='top', fontsize=8, rotation=-45)

    for xi, yi in zip(x, means_stretch):
        yi = round(yi, 4)
        plt.text(xi, yi, str(yi), ha='left', va='top', fontsize=8, rotation=-45)

    plt.xlabel('Number of Predicted Nodes/ # of operations')
    plt.ylabel('Error and Stretch')
    plt.title('# of operations vs Error/Stretch Graph | # of nodes: '+ total_number_of_nodes_in_graph + ' | Reps: ' + str(reps) + ' | Error ≤: ' + str(float(1/error_cutoff)))
    plt.grid(True)
    folder_name = "results"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # path_to_save = os.path.join('results', 'error_and_stretch_graphs', file_name)
    folder2 = "error_and_stretch_graphs_with_cut-off_AND_overlap"
    # folder2 = "error_and_stretch_graphs_with_cutoff"
    path_to_save = os.path.join('results', folder2, file_name)
    plt.legend(loc='best')
    plt.savefig(path_to_save)
    # plt.show()