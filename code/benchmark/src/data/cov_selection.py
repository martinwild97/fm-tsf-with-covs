import os
import shutil
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from src.data.dataset import DATA_DIR, Dataset


# --- Configuration ---
TIMESTAMP_COLUMN: str = "date"
MAX_LENGTH: int = 2048
MAX_LAG: int = 6
EVALUATION_METRIC: str = "MASE"
TEMP_MODEL_PATH: str = "./autogluon_temp_cov_selection"
OUTPUT_CSV_FILENAME: str = "covariate_lag_selection_results.csv"


def cov_selection() -> None:
    """
    Performs covariate and lag selection for time series forecasting using AutoGluon.

    This function reads sample data, iterates through each time series, and for each
    series, evaluates the performance of different covariates at various lag values.
    It trains an AutoGluon TimeSeriesPredictor for each covariate-lag combination
    and records the evaluation metric (MASE by default).

    The results, including dataset name, time series name, covariate name, lag,
    and the evaluation score, are saved to a CSV file.

    Raises
    ------
    FileNotFoundError
        If 'cov_selection_samples.csv' is not found in the specified DATA_DIR.
    """
    try:
        samples_df: pd.DataFrame = pd.read_csv(DATA_DIR / "cov_selection_samples.csv")
    except FileNotFoundError:
        print(f"Error: Could not find 'cov_selection_samples.csv' in {DATA_DIR}.")
        return

    all_results_list: List[Dict[str, Any]] = []

    for index, sample_row in samples_df.iterrows():
        dataset_name: str = sample_row["dataset_name"]
        ts_name: str = sample_row["ts_name"]
        print(f"\n\nProcessing Sample: Dataset '{dataset_name}', TS '{ts_name}'")

        ds: Dataset = Dataset(dataset_name)

        if ts_name != "_single_series_" and "ts_name" in ds.df.columns:
            df_for_sample: pd.DataFrame = ds.df[ds.df["ts_name"] == ts_name].copy()
        else:
            df_for_sample = ds.df.copy()

        df_current_ts: pd.DataFrame = df_for_sample[-MAX_LENGTH:].copy()

        fc_h: int = sample_row["fc_horizon"]

        # --- Train/Test Split ---
        test_data_length: int = int(len(df_current_ts) * 0.2)
        if test_data_length < fc_h:
            test_data_length = fc_h

        if len(df_current_ts) <= test_data_length:
            print(
                f"  Warning: Not enough data for meaningful train/test split for "
                f"{dataset_name}/{ts_name}. Length: {len(df_current_ts)}, "
                f"Test length required: {test_data_length}. Skipping this sample."
            )
            continue

        train_val_cutoff_idx: int = len(df_current_ts) - test_data_length
        train_val_df_full_loop: pd.DataFrame = df_current_ts.iloc[:train_val_cutoff_idx].copy()
        test_df_full_loop: pd.DataFrame = df_current_ts.iloc[train_val_cutoff_idx:].copy()

        print(f"  Dataset: {dataset_name}, TS: {ts_name}")
        print(f"  Total data for this TS (after MAX_LENGTH): {len(df_current_ts)} rows")
        print(f"  Training/Validation data (before lagging): {len(train_val_df_full_loop)} rows")
        print(f"  Test data (before lagging): {len(test_df_full_loop)} rows")

        # --- Main loop for covariate and lag selection ---
        current_sample_results: List[Dict[str, Any]] = []

        all_covariates_to_test: List[str] = list(set(ds.past_covariates + ds.future_covariates))
        if not all_covariates_to_test:
            print(
                f"  No covariates to test for {dataset_name}/{ts_name}. "
                "Skipping covariate loop."
            )
            continue

        target_columns_for_this_ds: List[str] = ds.targets
        if not target_columns_for_this_ds:
            print(
                f"  No target columns (ds.targets) defined for Dataset {dataset_name}. "
                "Skipping."
            )
            continue

        for cov_name in all_covariates_to_test:
            if cov_name not in df_current_ts.columns:
                print(
                    f"    Warning: Covariate '{cov_name}' not found in DataFrame for "
                    f"{dataset_name}/{ts_name}. Skipping this covariate."
                )
                continue

            print(f"\n  --- Evaluating Covariate: {cov_name} ---")
            for lag in range(0, MAX_LAG + 1):
                print(f"    Checking Lag: {lag}")

                current_lagged_feature_name: str = f"{cov_name}_lag{lag}"

                # Apply lag to training data
                train_val_df_current: pd.DataFrame = train_val_df_full_loop.copy()
                train_val_df_current[current_lagged_feature_name] = train_val_df_current[
                    cov_name
                ].shift(lag)

                # Apply lag to the full data before splitting for test set
                temp_full_df_for_lagging: pd.DataFrame = df_current_ts.copy()
                temp_full_df_for_lagging[current_lagged_feature_name] = temp_full_df_for_lagging[
                    cov_name
                ].shift(lag)
                test_df_current_with_lag: pd.DataFrame = temp_full_df_for_lagging.iloc[
                    train_val_cutoff_idx:
                ].copy()

                # Clean NaN values after lagging
                train_val_df_current_cleaned: pd.DataFrame = train_val_df_current.dropna(
                    subset=[current_lagged_feature_name] + target_columns_for_this_ds
                )
                test_df_current_with_lag_cleaned: pd.DataFrame = test_df_current_with_lag.dropna(
                    subset=[current_lagged_feature_name] + target_columns_for_this_ds
                )

                if train_val_df_current_cleaned.empty or len(train_val_df_current_cleaned) < 2 * fc_h:
                    print(
                        f"      Too few training data for {cov_name} with Lag {lag} after "
                        f"NaN removal ({len(train_val_df_current_cleaned)} rows). Skipping."
                    )
                    continue
                if test_df_current_with_lag_cleaned.empty or len(test_df_current_with_lag_cleaned) < fc_h:
                    print(
                        f"      Too few test data for {cov_name} with Lag {lag} after "
                        f"NaN removal ({len(test_df_current_with_lag_cleaned)} rows). Skipping."
                    )
                    continue

                # Prepare data for AutoGluon (melt multiple targets into 'unique_id' column)
                cols_for_melt_train: List[str] = [TIMESTAMP_COLUMN, current_lagged_feature_name] + target_columns_for_this_ds
                train_df_for_melt: pd.DataFrame = train_val_df_current_cleaned[cols_for_melt_train].copy()

                cols_for_melt_test: List[str] = [TIMESTAMP_COLUMN, current_lagged_feature_name] + target_columns_for_this_ds
                test_df_for_melt: pd.DataFrame = test_df_current_with_lag_cleaned[cols_for_melt_test].copy()

                melted_train_df: pd.DataFrame = train_df_for_melt.melt(
                    id_vars=[TIMESTAMP_COLUMN, current_lagged_feature_name],
                    value_vars=target_columns_for_this_ds,
                    var_name="unique_id",
                    value_name="y",
                )
                melted_test_df: pd.DataFrame = test_df_for_melt.melt(
                    id_vars=[TIMESTAMP_COLUMN, current_lagged_feature_name],
                    value_vars=target_columns_for_this_ds,
                    var_name="unique_id",
                    value_name="y",
                )

                train_data_for_autogluon: TimeSeriesDataFrame = TimeSeriesDataFrame.from_data_frame(
                    melted_train_df,
                    id_column="unique_id",
                    timestamp_column=TIMESTAMP_COLUMN,
                )
                test_data_for_autogluon: TimeSeriesDataFrame = TimeSeriesDataFrame.from_data_frame(
                    melted_test_df,
                    id_column="unique_id",
                    timestamp_column=TIMESTAMP_COLUMN,
                )

                if test_data_for_autogluon.empty:
                    print(
                        f"      Test data for AutoGluon is empty for {cov_name} with Lag {lag}. "
                        "Skipping."
                    )
                    continue

                if os.path.exists(TEMP_MODEL_PATH):
                    shutil.rmtree(TEMP_MODEL_PATH)

                predictor: TimeSeriesPredictor = TimeSeriesPredictor(
                    prediction_length=fc_h,
                    path=TEMP_MODEL_PATH,
                    target="y",
                    eval_metric=EVALUATION_METRIC,
                    known_covariates_names=[current_lagged_feature_name],
                    verbosity=0,
                )

                try:
                    predictor.fit(
                        train_data_for_autogluon, hyperparameters={"DirectTabular": {"GBM": {}}}
                    )
                    performance: Dict[str, float] = predictor.evaluate(test_data_for_autogluon)
                    score: float = performance[EVALUATION_METRIC]

                    current_sample_results.append(
                        {
                            "dataset_name": dataset_name,
                            "ts_name": ts_name,
                            "cov_name": cov_name,
                            "lag": lag,
                            EVALUATION_METRIC: score,
                        }
                    )
                    print(
                        f"      Covariate: {cov_name}, Lag: {lag}, {EVALUATION_METRIC}: "
                        f"{score:.4f}"
                    )

                except Exception as e:
                    print(f"      Error with {cov_name}, Lag {lag}: {e}")
                finally:
                    if os.path.exists(TEMP_MODEL_PATH):
                        shutil.rmtree(TEMP_MODEL_PATH)

        all_results_list.extend(current_sample_results)

    # --- Process and save results ---
    if all_results_list:
        results_df: pd.DataFrame = pd.DataFrame(all_results_list)

        cols_order: List[str] = ["dataset_name", "ts_name", "cov_name", "lag", EVALUATION_METRIC]
        results_df = results_df[cols_order]

        results_df.to_csv(OUTPUT_CSV_FILENAME, index=False)
        print(f"\nResults saved to '{OUTPUT_CSV_FILENAME}'.")
        print(results_df.head())
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    cov_selection()