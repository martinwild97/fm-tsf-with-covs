import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
TIMESERIES_INFO_CSV = DATA_DIR / "timeseries_info.csv"
TIMESERIES_DISTANCE_CSV = DATA_DIR / "time_series_distances.csv"

if __name__ == "__main__":
    # Load the CSV file
    df = pd.read_csv(TIMESERIES_INFO_CSV)

    # Define the list of features to use
    selected_features = [
        'entropy', 'lumpiness', 'stability', 'hurst', 'trend', 'spike', 'linearity',
        'curvature', 'e_acf1', 'seas_acf1',
        'diff1_acf1', 'diff2_acf1', 'seas_pacf',
        'diff1x_pacf5', 'diff2x_pacf5', 'nonlinearity',
        'unitroot_pp', 'unitroot_kpss', 'series_length',
        'diff1_acf10', 'diff2_acf10',
        'alpha', 'beta'
    ]

    # Filter the DataFrame to include only the selected features
    df_selected = df[['ts_id'] + selected_features].copy()
    df_selected = df_selected.set_index('ts_id')

    # Get all pairs of time series within the dataset
    series_ids = df_selected.index
    series_pairs = []
    for id1, id2 in combinations(series_ids, 2):
        series_pairs.append((id1, id2))

    total_pairs = len(series_pairs)
    print(f"Total pairs to process: {total_pairs}")

    num_features = len(selected_features)

    denominators = []
    for m in range(num_features):
        other_series_m = df_selected.iloc[:, m].dropna()
        if not other_series_m.empty:
            max_ck_m = other_series_m.max()
            min_ck_m = other_series_m.min()
            denominator = max_ck_m - min_ck_m
            if denominator == 0:
                raise ValueError(f"Denominator for feature {selected_features[m]} is zero.")
            denominators.append(denominator)

    results = []
    for i, (id1, id2) in enumerate(series_pairs):
        series1 = df_selected.loc[id1]
        series2 = df_selected.loc[id2]
        distance = 0
        for m in range(num_features):
            c_yi_m = series1.iloc[m]
            c_yj_m = series2.iloc[m]
            if pd.notna(c_yi_m) and pd.notna(c_yj_m):
                distance += abs(c_yi_m - c_yj_m) / denominator

        results.append({
            'ts_id_1': id1,
            'ts_id_2': id2,
            'distance': distance
        })
        # Update the progress bar
        if i % 1000 == 0:
            print(f"Processed {i} out of {total_pairs} pairs.")

    # Create a DataFrame from the results
    distance_df = pd.DataFrame(results)

    # Save the distance DataFrame to a CSV file
    distance_df.to_csv(TIMESERIES_DISTANCE_CSV, index=False)

    print("Distances between time series calculated and saved to time_series_distances.csv")