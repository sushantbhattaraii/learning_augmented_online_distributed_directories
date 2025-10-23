import pandas as pd
import os
import re

def save_error_stretch_to_excel(fractions, max_errors, min_errors, stretches, stretches_arrow, stretches_new, diameters_modified_mst, diameters_steiner_tree, file_name, reps, error_cutoff):
    """
    Save both raw and summary statistics of error and stretch data to an Excel file.
    """
    n_groups = len(fractions)
    # Group raw data by fraction
    groups_max_error = [max_errors[i::n_groups] for i in range(n_groups)]
    groups_min_error = [min_errors[i::n_groups] for i in range(n_groups)]
    groups_stretch = [stretches[i::n_groups] for i in range(n_groups)]
    groups_stretch_arrow = [stretches_arrow[i::n_groups] for i in range(n_groups)]
    groups_stretch_new = [stretches_new[i::n_groups] for i in range(n_groups)]
    groups_diameter_modified_mst = [diameters_modified_mst[i::n_groups] for i in range(n_groups)]
    groups_diameter_steiner_tree = [diameters_steiner_tree[i::n_groups] for i in range(n_groups)]

    # Prepare raw records
    total_nodes = int(re.findall(r'\d+\.?\d*', file_name)[0])
    raw_records = []
    for frac, max_err_list, min_err_list, str_list, str_arrow_list, str_new_list, diam_modified_mst_list, diam_steiner_tree_list in zip(fractions, groups_max_error, groups_min_error, groups_stretch, groups_stretch_arrow, groups_stretch_new, groups_diameter_modified_mst, groups_diameter_steiner_tree):
        num_nodes = int(total_nodes)
        for max_err, min_err, strc, str_arrow_c, str_new_c, diam_mod_c, diam_stei_c  in zip(max_err_list, min_err_list, str_list, str_arrow_list, str_new_list, diam_modified_mst_list, diam_steiner_tree_list):
            raw_records.append({
                'fraction': frac,
                'num_nodes': num_nodes,
                'max_error': max_err,
                'min_error': min_err,
                'stretch': strc,
                'stretch_arrow': str_arrow_c,
                'stretch_new': str_new_c,
                'diameter_modified_mst': diam_mod_c,
                'diameter_steiner_tree': diam_stei_c,
                'reps': reps,
                'error_cutoff': error_cutoff,
                'file_name': file_name
            })
    raw_df = pd.DataFrame(raw_records)

    # Compute summary statistics
    summary_records = []
    for frac, max_err_list, min_err_list, str_list, str_arrow_list in zip(fractions, groups_max_error, groups_max_error, groups_stretch, groups_stretch_arrow):
        num_nodes = int(frac * total_nodes)
        summary_records.append({
            'fraction': frac,
            'num_nodes': num_nodes,
            'mean_of_max_error': sum(max_err_list) / len(max_err_list),
            'min_of_max_error': min(max_err_list),
            'max_of_max_error': max(max_err_list),
            'mean_of_min_error': sum(min_err_list) / len(min_err_list),
            'min_of_min_error': min(min_err_list),
            'max_of_min_error': max(min_err_list),
            'mean_stretch': sum(str_list) / len(str_list),
            'min_stretch': min(str_list),
            'max_stretch': max(str_list),
            'mean_stretch_arrow': sum(str_arrow_list) / len(str_arrow_list),
            'min_stretch_arrow': min(str_arrow_list),
            'max_stretch_arrow': max(str_arrow_list),
            'reps': reps,
            'error_cutoff': error_cutoff,
            'file_name': file_name
        })
    summary_df = pd.DataFrame(summary_records)

    # Ensure output directory exists
    folder = "results/threshEqualsDiamOver3_data/random"
    os.makedirs(folder, exist_ok=True)
    excel_path = os.path.join(folder, f"{os.path.splitext(file_name)[0]}.xlsx")

    # Write both raw and summary to Excel
    with pd.ExcelWriter(excel_path) as writer:
        raw_df.to_excel(writer, sheet_name='raw', index=False)
        summary_df.to_excel(writer, sheet_name='summary', index=False)

    return excel_path

# Example usage within your plotting function:
# excel_file = save_error_stretch_to_excel(fractions, errors, stretches, file_name, reps, error_cutoff)
# print(f"Data saved to {excel_file}")
