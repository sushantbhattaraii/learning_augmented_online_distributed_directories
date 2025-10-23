import re
import subprocess
from plot_graph import *
import argparse
import os
import re
from fractions import Fraction
from save_data_to_excel import *

def main(network_file_name, repetitions, error_cutoff, overlap):
    max_errors = []
    min_errors = []
    stretches = []
    stretches_arrow = []
    stretches_new = []
    diameters_modified_mst = []
    diameters_steiner_tree = []
        
    nodes_count = [] 

    nodes_num = int(re.findall(r"\d+", network_file_name)[0])
    # print("Total nodes in the graph: ", nodes_num)
    # print("Type of n: ", type(nodes_num))
        
    fractions = []
    while nodes_num > 1:
        nodes_num //=2
        fractions.append(nodes_num)

    fractions = fractions[::-1]

    for rep in range(repetitions):
        # The four fraction values
        # fractions = [n/2, n/4, n/8, n/16, n/32, n/64, n/128]
        # fractions = [10, 100, 1000, 10000]
        
        # print("Fractions to be used: ", fractions)

        pattern = re.compile(
            r"Overall max error \(max_i\(distance_in_G / diameter_G\)\) =\s*([0-9.+\-eE]+)"
        )

        pattern4 = re.compile(
            r"Overall min error \(min_i\(distance_in_G / diameter_G\)\) =\s*([0-9.+\-eE]+)"
        )

        pattern2 = re.compile(
            r"Total number of vertices \(n\):\s*([0-9.+\-eE]+)"
        )

        pattern3 = re.compile(
            r"Stretch \(sum_of_distance_in_T / sum_of_distance_in_G\) =\s*([0-9.+\-eE]+)"
        )

        pattern5 = re.compile(
            r"Stretch_Arrow \(sum_of_distance_in_mst_g / sum_of_distance_in_G\) =\s*([0-9.+\-eE]+)"
        )

        pattern8 = re.compile(
            r"Stretch_New \(sum_of_distance_in_T_new / sum_of_distance_in_G\) =\s*([0-9.+\-eE]+)"
        )

        pattern6 = re.compile(
            r"Diameter of modified MST =\s*([0-9.+\-eE]+)"
        )

        pattern7 = re.compile(
            r"Diameter of Steiner tree =\s*([0-9.+\-eE]+)"
        )

        for frac in fractions:
            print(f"\n=== Running run.py with --operations {frac}  and Iteration {rep} ===")
            proc = subprocess.Popen(
                ["python", "run.py", "--fraction", str(frac), "--network", str(network_file_name), "--cutoff", str(error_cutoff), "--overlap", str(overlap)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            captured = ""

            # read line by line
            for line in proc.stdout:
                print(line, end="")       # echo to your terminal
                captured += line
                m = pattern.search(line)
                m2 = pattern2.search(line)
                m3 = pattern3.search(line)
                m4 = pattern4.search(line)
                m5 = pattern5.search(line)
                m6 = pattern6.search(line)
                m7 = pattern7.search(line)
                m8 = pattern8.search(line)
                if m:
                    max_error_value = float(m.group(1))
                if m2:
                    num_nodes = int(m2.group(1))
                if m3:
                    stretch_value = float(m3.group(1))
                if m4:
                    min_error_value = float(m4.group(1))
                if m5:
                    stretch_arrow_value = float(m5.group(1))
                if m8:
                    stretch_new_value = float(m8.group(1))
                if m6:
                    diameter_of_modified_mst = int(m6.group(1))
                if m7:
                    diameter_of_steiner_tree = int(m7.group(1))
            
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, proc.args)

            max_errors.append(max_error_value)
            min_errors.append(min_error_value)
            stretches.append(stretch_value)
            stretches_arrow.append(stretch_arrow_value)
            stretches_new.append(stretch_new_value)
            diameters_modified_mst.append(diameter_of_modified_mst)
            diameters_steiner_tree.append(diameter_of_steiner_tree)
            nodes_count.append(num_nodes)


    # Searching for the diameter of the graph from its filename
    md = re.search(r"diameter(\d+)", str(network_file_name))
    if md:
        diameter_value = md.group(1)   # this is the string "48"

    

    print("Plotting the error graph...")

    # filename = str(nodes_count[0])+"nodes_diameter"+str(diameter_value)+"_cutoff"+str(error_cutoff)+"-repetitions"+str(repetitions)+".png"
    filename = str(nodes_count[0])+"nodes_diameter"+str(diameter_value)+"_cutoff"+str(error_cutoff)+"-repetitions"+str(repetitions)+ "-overlap"+str(overlap)+".png"

    # plot_error_and_stretch_graph_with_boxplot(fractions, errors, filename, repetitions, stretches, error_cutoff, overlap)
    save_error_stretch_to_excel(fractions, max_errors, min_errors, stretches, stretches_arrow, stretches_new, diameters_modified_mst, diameters_steiner_tree, filename, repetitions, error_cutoff)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Running the experiment with different fractions of predicted nodes and with different graphs... ")
    p.add_argument(
        "-n",
        "--network",
        required=True,
        type=str,
        help="The network file name to run an algorithm on (e.g. '256random_diameter71test.edgelist')"
    )
    p.add_argument(
        "-r",
        "--repetitions",
        default=1,
        type=int,
        help="Number of repetitions for the experiment (default: 1)"
    )
    p.add_argument(
        "-c",
        "--cutoff",
        default=1.0,
        type=float,
        help="Cutoff parameter for the error value (implies the error value cannot go beyond this cutoff)"
    )
    p.add_argument(
        "-o",
        "--overlap",
        default=100,
        type=int,
        help="Overlap of the actual nodes requesting for the object (in percentage)"
    )
    args = p.parse_args()
    main(args.network, args.repetitions, args.cutoff, args.overlap)
