import pandas as pd
import numpy as np
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
TIMESERIES_INFO_CSV = DATA_DIR / "timeseries_info.csv"
TIMESERIES_DISTANCE_CSV = DATA_DIR / "time_series_distances.csv"

def greedy_fps_selection(
    ts_info_path: str = TIMESERIES_INFO_CSV,
    ts_distance_path: str = TIMESERIES_DISTANCE_CSV,
    target_size: int = 150,
    min_samples_per_dataset: int = 5
) -> list:
    """
    Selects a diverse subset of time series using Greedy Farthest-Point Sampling (FPS)
    while ensuring representation from all original datasets. In addition,
    ensures that each dataset contributes at least min_samples_per_dataset time series,
    if possible.

    Args:
        ts_info_path: Path to the CSV file containing time series info (ts_id, dataset_name).
        ts_distance_path: Path to the CSV file containing pairwise distances
                          (ts_id_1, ts_id_2, distance).
        target_size: The desired number of time series in the final benchmark set.
        min_samples_per_dataset: Minimum number of time series to select from each dataset.
    
    Returns:
        A sorted list of selected ts_id values.
    """
    print("Loading data...")
    # Load time series info to get dataset mapping and total count
    ts_info = pd.read_csv(ts_info_path)
    num_total_ts = ts_info['ts_id'].max() + 1
    ts_id_to_dataset = ts_info.set_index('ts_id')['dataset_name']
    all_dataset_names = set(ts_info['dataset_name'].unique())
    num_unique_datasets = len(all_dataset_names)
    
    print(f"Found {num_total_ts} time series across {num_unique_datasets} datasets.")

    start_time = time.time()
    # Initialize distance matrix with NaNs and set diagonal to 0
    distance_matrix = np.full((num_total_ts, num_total_ts), np.nan, dtype=np.float32)
    np.fill_diagonal(distance_matrix, 0)

    # Read distances and fill in the symmetric distance matrix
    dist_df = pd.read_csv(ts_distance_path)
    for _, row in dist_df.iterrows():
        id1, id2, dist = int(row['ts_id_1']), int(row['ts_id_2']), float(row['distance'])
        if 0 <= id1 < num_total_ts and 0 <= id2 < num_total_ts:
             distance_matrix[id1, id2] = dist
             distance_matrix[id2, id1] = dist  # Assuming symmetric distance
        else:
             print(f"Warning: Found out-of-bounds index in distance file: {id1}, {id2}")

    end_time = time.time()
    print(f"Distance matrix built in {end_time - start_time:.2f} seconds.")

    # --- FPS Algorithm ---
    print("Starting Greedy FPS selection...")
    selected_ts_ids = set()
    # Initialize the counts of selected points per dataset
    dataset_counts = {dataset: 0 for dataset in all_dataset_names}

    # 1. Initialization: Select the first point based on the maximum total distance
    sum_distances = np.sum(distance_matrix, axis=1)
    first_ts_id = np.argmax(sum_distances)
    selected_ts_ids.add(first_ts_id)
    dataset = ts_id_to_dataset[first_ts_id]
    dataset_counts[dataset] += 1
    print(f"Selected initial point: {first_ts_id} (Dataset: {dataset})")

    # Initialize minimum distances from all points to the selected set S
    min_distances_to_S = distance_matrix[:, first_ts_id].copy()
    min_distances_to_S[first_ts_id] = -np.inf

    start_time = time.time()
    # 2. Iteration: Select remaining points until target size is reached.
    while len(selected_ts_ids) < target_size:
        current_selection_count = len(selected_ts_ids)
        
        # Identify datasets that are still underrepresented:
        underrepresented = {d for d, count in dataset_counts.items() if count < min_samples_per_dataset}

        candidate_indices = np.arange(num_total_ts)
        valid_candidates_mask = (min_distances_to_S > -np.inf)

        if underrepresented:
            # Prioritize candidates from underrepresented datasets
            underrep_mask = ts_id_to_dataset.isin(underrepresented).values
            candidate_mask = valid_candidates_mask & underrep_mask

            # If no valid candidate from underrepresented datasets is found, fall back to all valid candidates
            if not np.any(candidate_mask):
                 print("Warning: No available candidate from underrepresented datasets; using all valid candidates.")
                 candidate_mask = valid_candidates_mask
        else:
            candidate_mask = valid_candidates_mask

        candidate_indices_filtered = candidate_indices[candidate_mask]
        if len(candidate_indices_filtered) == 0:
             print("Warning: No more valid candidates to select. Stopping early.")
             break

        min_dists_filtered = min_distances_to_S[candidate_mask]
        best_candidate_local_idx = np.argmax(min_dists_filtered)
        next_ts_id = candidate_indices_filtered[best_candidate_local_idx]

        selected_ts_ids.add(next_ts_id)
        dataset = ts_id_to_dataset[next_ts_id]
        dataset_counts[dataset] += 1

        # Mark the candidate as selected
        min_distances_to_S[next_ts_id] = -np.inf

        # Update the minimum distances from all other points to the newly updated selected set
        distances_to_new = distance_matrix[:, next_ts_id]
        min_distances_to_S = np.minimum(min_distances_to_S, distances_to_new)

        if (current_selection_count + 1) % 20 == 0 or (current_selection_count + 1) == target_size:
             print(f"Selected {current_selection_count + 1}/{target_size} points. "
                   f"Last added: {next_ts_id} (Dataset: {dataset})")

    end_time = time.time()
    print(f"FPS selection finished in {end_time - start_time:.2f} seconds.")

    # Final check: report dataset coverage
    final_dataset_counts = {}
    for ts_id in selected_ts_ids:
        ds = ts_id_to_dataset[ts_id]
        final_dataset_counts[ds] = final_dataset_counts.get(ds, 0) + 1

    missing_minimum = {ds: min_samples_per_dataset - count
                       for ds, count in final_dataset_counts.items() if count < min_samples_per_dataset}
    if missing_minimum:
        print("\nWarning: The following datasets did not reach the minimum required samples:")
        for ds, deficit in missing_minimum.items():
            print(f"  {ds}: missing {deficit} time series")
    else:
        print(f"\nSuccessfully selected {len(selected_ts_ids)} time series with "
              f"at least {min_samples_per_dataset} samples per dataset.")

    return sorted(list(selected_ts_ids))


if __name__ == "__main__":
    selection = greedy_fps_selection()
    print(f"\nSelected Time Series IDs ({len(selection)}):")
    print(','.join(map(str, selection)))
