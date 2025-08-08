import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.time_feature.lag import get_lags_for_frequency

from src.data.dataset import DATA_DIR, Dataset
from src.helper_functions import extract_base_frequency
from src.models.schemas import ModelInput

# --- Path Configuration ---
# Resolving paths relative to the current file for robustness.
_CURRENT_DIR: Path = Path(__file__).resolve().parent
DATASET_INFO_CSV: Path = DATA_DIR / "dataset_info.csv"
DATASET_DIR: Path = DATA_DIR / "prepared_datasets"
TIMESERIES_INFO_CSV: Path = DATA_DIR / "timeseries_info.csv"
SAMPLE_INFO_CSV: Path = DATA_DIR / "sample_info.csv"
COV_LAG_SELECTION_CSV: Path = DATA_DIR / "covariate_lag_selection_results.csv"

# --- Constants ---
# Defines the forecast horizons for different time granularities.
FC_HORIZONS: Dict[str, List[int]] = {
    "h": [24, 168],  # Hourly: 24 hours, 168 hours (1 week)
    "D": [7, 90],  # Daily: 7 days (1 week), 90 days (approx. 3 months)
    "W": [26],  # Weekly: 26 weeks (approx. 6 months)
    "M": [12],  # Monthly: 12 months (1 year)
    "Q": [8],  # Quarterly: 8 quarters (2 years)
    "Y": [5],  # Yearly: 5 years
}

# Defines the valid modes for covariate selection/handling.
COV_MODES: List[str] = [
    "all",
    "no",
    "noise",
    "only_past",
    "only_future",
    "time",
    "lagged_target",
    "selected",
    "lagged_selected",
]

# --- Logger Setup ---
# Initializes a logger specific to data loading operations.
ds_logger: logging.Logger = logging.getLogger("DataLoader")


class TSInfo:
    """
    Represents time series metadata, loaded and filtered from a CSV file.

    Parameters
    ----------
    path : Path, default=TIMESERIES_INFO_CSV
        The file path to the CSV containing time series information.
    ts_ids : Optional[List[int]], default=None
        A list of time series IDs to filter the data by.
    dataset_names : Optional[List[str]], default=None
        A list of dataset names to filter the data by.
    target_names : Optional[List[str]], default=None
        A list of target variable names to filter the data by.
    ts_names : Optional[List[str]], default=None
        A list of time series specific names to filter the data by.

    Attributes
    ----------
    df : pd.DataFrame
        The filtered DataFrame containing time series information.
    ts_ids : List[int]
        Unique list of time series IDs in the filtered DataFrame.
    dataset_names : List[str]
        Unique list of dataset names in the filtered DataFrame.
    target_names : List[str]
        Unique list of target names in the filtered DataFrame.
    ts_names : List[str]
        Unique list of time series specific names in the filtered DataFrame.

    Raises
    ------
    SystemExit
        If the timeseries info file is not found or an error occurs during processing.
    """

    def __init__(
        self,
        path: Path = TIMESERIES_INFO_CSV,
        ts_ids: Optional[List[int]] = None,
        dataset_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
        ts_names: Optional[List[str]] = None,
    ) -> None:
        self.columns: List[str] = ["ts_id", "dataset_name", "target_name", "ts_name"]
        self._filter_ts_ids = ts_ids
        self._filter_dataset_names = dataset_names
        self._filter_target_names = target_names
        self._filter_ts_names = ts_names

        if not path.exists():
            ds_logger.error(f"Timeseries info file not found at {path}")
            sys.exit(1)
        else:
            try:
                self.df: pd.DataFrame = pd.read_csv(path)
                self.df = self.df[self.columns]

                # Type casting for consistency.
                self.df["ts_id"] = self.df["ts_id"].astype(int)
                for col in ["dataset_name", "target_name", "ts_name"]:
                    self.df[col] = self.df[col].fillna("").astype(str)

                # Filtering the DataFrame based on constructor arguments.
                for col in self.columns:
                    filter_values = getattr(self, f"_filter_{col}s")
                    if filter_values is not None:
                        self.df = self.df[self.df[col].isin(filter_values)]

            except Exception as e:
                ds_logger.error(f"Error processing timeseries info file at {path}: {e}")
                sys.exit(1)

        # Assign unique values from filtered DataFrame to instance attributes.
        for col in self.columns:
            setattr(self, f"{col}s", self.df[col].unique().tolist())


class Sample:
    """
    Represents a data sample for forecasting, including dataset details,
    forecast horizon, and a specific data window.

    Parameters
    ----------
    sample_id : int
        A unique identifier for the sample.
    ts_ids : List[int]
        A list of time series IDs included in this sample.
    fc_horizon : int
        The forecast horizon (number of steps to predict).
    window : int
        The window index, used for defining different training/test splits.

    Attributes
    ----------
    id : int
        The unique identifier for the sample.
    ts_ids : List[int]
        List of time series IDs associated with this sample.
    dataset_name : str
        The name of the dataset this sample belongs to.
    ts_name : str
        The specific time series name (e.g., '_single_series_' or actual TS name).
    targets : List[str]
        List of target variable names for this sample.
    num_ts : int
        The number of individual time series within this sample.
    fc_horizon : int
        The forecast horizon for this sample.
    window : int
        The window index for this sample.
    """

    def __init__(self, sample_id: int, ts_ids: List[int], fc_horizon: int, window: int) -> None:
        self.id: int = sample_id
        self.ts_ids: List[int] = ts_ids
        ts_info: TSInfo = TSInfo(ts_ids=ts_ids)
        self.dataset_name: str = ts_info.dataset_names[0]
        self.ts_name: str = ts_info.ts_names[0]
        self.targets: List[str] = ts_info.target_names
        self.num_ts: int = len(self.ts_ids)
        self.fc_horizon: int = fc_horizon
        self.window: int = window

    def __repr__(self) -> str:
        return (
            f"Sample(dataset_name={self.dataset_name}, ts_name={self.ts_name}, "
            f"targets={self.targets}, fc_horizon={self.fc_horizon}, window={self.window})"
        )


def prepare_samples(
    ts_ids: List[int],
    fc_horizons: Dict[str, List[int]] = FC_HORIZONS,
    windows_per_ts: Optional[int] = 3,
    sample_info_path: Optional[Path] = SAMPLE_INFO_CSV,
    save: Optional[bool] = True,
) -> None:
    """
    Prepares a list of `Sample` objects based on provided time series IDs,
    forecast horizons, and the number of windows per time series.

    This function iterates through datasets and time series, generating
    a `Sample` object for each combination of forecast horizon and window.
    If `save` is True, it also writes the sample information to a CSV file.

    Parameters
    ----------
    ts_ids : List[int]
        A list of time series IDs to include in the samples.
    fc_horizons : Dict[str, List[int]], default=FC_HORIZONS
        A dictionary mapping base granularity (e.g., 'h', 'D') to a list of
        forecast horizons.
    windows_per_ts : Optional[int], default=3
        The number of historical windows to create for each time series and
        forecast horizon combination.
    sample_info_path : Optional[Path], default=SAMPLE_INFO_CSV
        The file path to save the sample information CSV.
    save : Optional[bool], default=True
        If True, saves the generated sample information to a CSV file.
    """
    samples: List[Sample] = []
    ts_info_all: TSInfo = TSInfo(ts_ids=ts_ids)

    for ds_name in ts_info_all.dataset_names:
        ds: Dataset = Dataset(ds_name)
        base_granularity: str = extract_base_frequency(ds.granularity)
        
        # Get TS names relevant to the current dataset and overall ts_ids filter
        current_ds_ts_names: List[str] = TSInfo(
            ts_ids=ts_ids, dataset_names=[ds_name]
        ).ts_names

        for fc_horizon in fc_horizons.get(base_granularity, []):
            for window in range(windows_per_ts):
                if len(current_ds_ts_names) > 1:
                    for ts_name in current_ds_ts_names:
                        # Filter ts_ids further for the specific ts_name within the dataset
                        filtered_ts_ids: List[int] = TSInfo(
                            ts_ids=ts_ids, dataset_names=[ds_name], ts_names=[ts_name]
                        ).ts_ids
                        sample_id_counter: int = len(samples) # Assign ID before appending
                        sample: Sample = Sample(sample_id_counter, filtered_ts_ids, fc_horizon, window)
                        samples.append(sample)
                else:
                    # If only one time series name or '_single_series_', process it directly
                    filtered_ts_ids: List[int] = TSInfo(
                        ts_ids=ts_ids, dataset_names=[ds_name]
                    ).ts_ids
                    sample_id_counter: int = len(samples) # Assign ID before appending
                    sample: Sample = Sample(sample_id_counter, filtered_ts_ids, fc_horizon, window)
                    samples.append(sample)

    if save:
        try:
            with open(sample_info_path, "w") as f:
                f.write("sample_id,dataset_name,ts_name,ts_ids,targets,num_ts,fc_horizon,window\n")
                for sample_obj in samples:
                    ts_ids_str: str = ",".join(map(str, sample_obj.ts_ids))
                    targets_str: str = ",".join(sample_obj.targets)
                    f.write(
                        f"{sample_obj.id},{sample_obj.dataset_name},{sample_obj.ts_name},"
                        f"{ts_ids_str},{targets_str},{sample_obj.num_ts},{sample_obj.fc_horizon},"
                        f"{sample_obj.window}\n"
                    )
            ds_logger.info(f"Successfully saved samples to {sample_info_path}")
        except IOError as e:
            ds_logger.error(f"Error saving samples to {sample_info_path}: {e}")


class DataLoader:
    """
    Loads and prepares time series data for model input based on various
    covariate modes and sample configurations.

    This class acts as an iterable, yielding `ModelInput` objects for each
    sample defined in the `sample_info_csv`.

    Parameters
    ----------
    sample_info_csv : Optional[Path], default=SAMPLE_INFO_CSV
        Path to the CSV file containing sample information.
    mode : Optional[str], default="all"
        Defines how covariates are handled.
        Options are: "all", "no", "noise", "only_past", "only_future", "time",
        "lagged_target", "selected", "lagged_selected".
    quantile : Optional[float], default=0.9
        The quantile value for model input (e.g., for probabilistic forecasts).
    max_ts_length : Optional[int], default=None
        Maximum length to truncate the time series data to.

    Attributes
    ----------
    samples : List[Sample]
        A list of `Sample` objects loaded from `sample_info_csv`.
    ts_ids : List[int]
        Unique time series IDs present in the loaded samples.
    ts_info : TSInfo
        TSInfo object containing metadata for all relevant time series.
    cov_lag_selection : Optional[pd.DataFrame]
        DataFrame containing results from covariate lag selection, if available.
    mode : str
        The current covariate handling mode.
    quantile : float
        The quantile value for model input.
    max_ts_length : Optional[int]
        The maximum time series length.
    current_idx : int
        Internal counter for iteration.

    Raises
    ------
    FileNotFoundError
        If `sample_info_csv` or `COV_LAG_SELECTION_CSV` are not found.
    ValueError
        If an invalid mode is provided during initialization or `change_mode`.
    """

    def __init__(
        self,
        sample_info_csv: Optional[Path] = SAMPLE_INFO_CSV,
        mode: Optional[str] = "all",
        quantile: Optional[float] = 0.9,
        max_ts_length: Optional[int] = None,
    ) -> None:
        self.change_mode(mode)
        self.quantile: float = quantile
        self.max_ts_length: Optional[int] = max_ts_length
        self.current_idx: int = 0
        self.samples: List[Sample] = []
        self.ts_ids: List[int] = []
        self.ts_info: Optional[TSInfo] = None
        self.cov_lag_selection: Optional[pd.DataFrame] = None

        # Load Sample Info
        unique_ts_ids: set[int] = set()
        try:
            df_samples: pd.DataFrame = pd.read_csv(sample_info_csv)
            for _, row in df_samples.iterrows():
                ts_ids_list: List[int] = list(map(int, row["ts_ids"].split(",")))
                unique_ts_ids.update(ts_ids_list)
                sample: Sample = Sample(
                    row["sample_id"], ts_ids_list, int(row["fc_horizon"]), int(row["window"])
                )
                self.samples.append(sample)
            self._num_samples: int = len(self.samples)
            self.ts_ids = list(unique_ts_ids)
            self.ts_info = TSInfo(ts_ids=self.ts_ids)
        except FileNotFoundError:
            ds_logger.error(f"Sample info file not found at {sample_info_csv}.")
            raise
        except Exception as e:
            ds_logger.error(f"Error loading samples from {sample_info_csv}: {e}")
            raise

        # Load Covariate Lag Selection results
        try:
            self.cov_lag_selection = pd.read_csv(COV_LAG_SELECTION_CSV)
        except FileNotFoundError:
            ds_logger.warning(
                f"Covariate lag selection results file not found at {COV_LAG_SELECTION_CSV}. "
                "The 'selected' and 'lagged_selected' modes will not function as expected."
            )
            self.cov_lag_selection = None
        except Exception as e:
            ds_logger.error(
                f"Error loading covariate lag selection results from {COV_LAG_SELECTION_CSV}: {e}"
            )
            self.cov_lag_selection = None

    def __iter__(self) -> "DataLoader":
        """
        Initializes the iterator for the DataLoader.

        Returns
        -------
        DataLoader
            The DataLoader instance itself.
        """
        self.current_idx = 0
        return self

    def __next__(self) -> ModelInput:
        """
        Retrieves the next `ModelInput` object in the iteration.

        Returns
        -------
        ModelInput
            The `ModelInput` object for the current sample.

        Raises
        ------
        StopIteration
            When all samples have been processed.
        """
        if self.current_idx >= self._num_samples:
            raise StopIteration

        model_input: ModelInput = self.get_model_input(current_idx=self.current_idx)

        self.current_idx += 1
        return model_input

    def __len__(self) -> int:
        """
        Returns the total number of samples loaded.

        Returns
        -------
        int
            The total number of samples.
        """
        return self._num_samples

    def _add_time_features(self, df: pd.DataFrame, granularity: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Adds time-based features to the DataFrame using GluonTS utilities.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing a 'date' column.
        granularity : str
            The frequency string of the time series (e.g., 'h', 'D', 'W').

        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            A tuple containing:
            - The DataFrame with added time features.
            - A list of names of the newly added time feature columns.
        """
        # Ensure 'date' column is datetime type for PeriodIndex conversion.
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        period_index: pd.PeriodIndex = pd.PeriodIndex(
            df["date"], freq=extract_base_frequency(granularity)
        )
        time_features_list: List[Any] = time_features_from_frequency_str(granularity)

        feature_names: List[str] = []
        for time_feature_func in time_features_list:
            feature_name: str = f"time_feature_{time_feature_func.__name__}"
            feature_names.append(feature_name)
            try:
                df[feature_name] = time_feature_func(period_index)
            except Exception as e:
                ds_logger.error(
                    f"Error generating time feature {feature_name} with function "
                    f"{time_feature_func.__name__}: {e}"
                )
                if feature_name in df.columns:
                    df.drop(columns=[feature_name], inplace=True)
                    feature_names.remove(feature_name)
        return df, feature_names

    def _cov_selection(
        self, sample: Sample, df: pd.DataFrame, lagged: bool = False, max_num_covs: int = 3
    ) -> Tuple[List[str], Optional[pd.DataFrame]]:
        """
        Selects covariates based on pre-calculated selection results (e.g., MASE scores).

        Parameters
        ----------
        sample : Sample
            The current `Sample` object.
        df : pd.DataFrame
            The DataFrame containing the time series data.
        lagged : bool, default=False
            If True, applies the best lag found for selected covariates.
        max_num_covs : int, default=3
            The maximum number of top covariates to select.

        Returns
        -------
        Tuple[List[str], Optional[pd.DataFrame]]
            A tuple containing:
            - A list of selected covariate names.
            - The modified DataFrame with selected/lagged covariates, or None if an error occurs.
        """
        if self.cov_lag_selection is None:
            ds_logger.error(
                "Covariate lag selection results not loaded. Cannot perform covariate selection."
            )
            return [], None

        cov_lag_df: pd.DataFrame = self.cov_lag_selection[
            (self.cov_lag_selection["dataset_name"] == sample.dataset_name)
            & (self.cov_lag_selection["ts_name"] == sample.ts_name)
        ]

        if cov_lag_df.empty:
            ds_logger.warning(f"No matching features found for sample {sample.id}. Returning empty lists.")
            return [], df
        
        # Sort by evaluation metric (e.g., MASE) to get best performing covariates.
        cov_lag_df = cov_lag_df.sort_values(by=["MASE"], ascending=True)

        selected_covs: List[str] = []
        modified_df: Optional[pd.DataFrame] = df.copy()

        if lagged:
            # Select unique covariates based on their best lag, then take top 'max_num_covs'
            cov_lag_df = cov_lag_df.drop_duplicates(subset=["cov_name"], keep="first").head(
                max_num_covs
            )
            selected_covs = cov_lag_df["cov_name"].tolist()

            for cov in selected_covs:
                matching_row = cov_lag_df[cov_lag_df["cov_name"] == cov]
                if not matching_row.empty and cov in modified_df.columns:
                    lag_value: int = matching_row["lag"].values[0]
                    modified_df[cov] = modified_df[cov].shift(lag_value)
                elif cov not in modified_df.columns:
                    ds_logger.warning(
                        f"Covariate '{cov}' not found in DataFrame for lagging. Skipping."
                    )
            # Drop rows with NaN values introduced by shifting
            if modified_df is not None:
                modified_df.dropna(inplace=True)
                modified_df.reset_index(drop=True, inplace=True)
        else:
            # Select top 'max_num_covs' covariates with lag 0 (no lag applied)
            cov_lag_df = cov_lag_df[cov_lag_df["lag"] == 0].head(max_num_covs)
            selected_covs = cov_lag_df["cov_name"].tolist()
            # No shifting needed for lag 0, just ensuring these are the only covariates kept.

        return selected_covs, modified_df

    def get_sample(self, sample_id: int) -> Optional[Sample]:
        """
        Retrieves a `Sample` object by its ID.

        Parameters
        ----------
        sample_id : int
            ID of the sample to retrieve.

        Returns
        -------
        Optional[Sample]
            The `Sample` object if found, otherwise None.
        """
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        ds_logger.error(f"Sample with ID {sample_id} not found.")
        return None

    def change_mode(self, mode: str) -> None:
        """
        Changes the covariate handling mode of the DataLoader.

        Parameters
        ----------
        mode : str
            The new mode to set. Must be one of `COV_MODES`.

        Raises
        ------
        ValueError
            If an invalid mode is provided.
        """
        if mode not in COV_MODES:
            ds_logger.error(
                f"Invalid mode: '{mode}'. Available modes are: {COV_MODES}. "
                "Setting mode to 'all' as a fallback."
            )
            self.mode = "all"
        else:
            self.mode = mode

    def set_max_ts_length(self, max_ts_length: int) -> None:
        """
        Sets the maximum time series length for data truncation.

        Parameters
        ----------
        max_ts_length : int
            The maximum length of the time series. Must be greater than 0.

        Raises
        ------
        ValueError
            If `max_ts_length` is not greater than 0.
        """
        if max_ts_length is not None and max_ts_length <= 0:
            ds_logger.error(f"Invalid max_ts_length: {max_ts_length}. Must be greater than 0.")
            raise ValueError("max_ts_length must be greater than 0.")
        self.max_ts_length = max_ts_length

    def get_model_input(
        self, sample_id: Optional[int] = None, current_idx: Optional[int] = None, mode: Optional[str] = None
    ) -> Optional[ModelInput]:
        """
        Prepares and returns a `ModelInput` object for a specified sample.

        This method applies filtering, truncation, and covariate handling
        based on the DataLoader's configuration and the specified mode.

        Parameters
        ----------
        sample_id : Optional[int], default=None
            The ID of the sample to retrieve.
        current_idx : Optional[int], default=None
            The current index in the internal sample list (used during iteration).
        mode : Optional[str], default=None
            Overrides the DataLoader's current mode for this specific call.

        Returns
        -------
        Optional[ModelInput]
            A `ModelInput` object ready for model training/prediction, or None
            if the sample cannot be processed or found.

        Raises
        ------
        ValueError
            If neither `sample_id` nor `current_idx` is provided, or both are.
            If `max_ts_length` is invalid.
        IndexError
            If `current_idx` is out of bounds.
        """
        if (sample_id is None and current_idx is None) or (
            sample_id is not None and current_idx is not None
        ):
            ds_logger.error("Either sample_id or current_idx must be provided, but not both.")
            raise ValueError("Either sample_id or current_idx must be provided.")

        sample: Optional[Sample] = None
        if sample_id is not None:
            sample = self.get_sample(sample_id)
            if sample is None:
                # Error already logged by get_sample
                return None
        elif current_idx is not None:
            if not (0 <= current_idx < self._num_samples):
                ds_logger.error(
                    f"Sample index {current_idx} out of range. "
                    f"Must be between 0 and {self._num_samples - 1}."
                )
                raise IndexError("Sample index out of range.")
            sample = self.samples[current_idx]

        if mode is not None:
            self.change_mode(mode)

        if sample is None:  # Should not happen given checks above, but for type safety.
            return None

        dataset: Dataset = Dataset(sample.dataset_name)
        df: pd.DataFrame = dataset.df.copy()

        # Filter the dataset based on the time series name
        if sample.ts_name != "_single_series_" and "ts_name" in df.columns:
            df = df[df["ts_name"] == sample.ts_name].copy()
            df.drop(columns=["ts_name"], inplace=True)

        # Filter the dataset based on the target names
        if dataset.num_targets > len(sample.targets):
            targets_to_drop: List[str] = [
                target for target in dataset.targets if target not in sample.targets
            ]
            df = df.drop(columns=targets_to_drop)

        # Truncate the dataset for forecast horizon * window
        if sample.window > 0:
            df = df.iloc[: -(sample.fc_horizon * sample.window)]

        all_covs: List[str] = dataset.past_covariates + dataset.future_covariates
        past_covs: List[str] = []
        future_covs: List[str] = []
        categorical_covs: List[str] = []

        match self.mode:
            case "all":
                past_covs = dataset.past_covariates
                future_covs = dataset.future_covariates
                categorical_covs = dataset.categorical_covariates
            case "no":
                if all_covs:  # Check if list is not empty to avoid unnecessary drop
                    df.drop(columns=all_covs, inplace=True)
            case "noise":
                for cov in all_covs:
                    if cov in df.columns:
                        df[cov] = np.random.permutation(df[cov])
                past_covs = dataset.past_covariates
                future_covs = dataset.future_covariates
                categorical_covs = dataset.categorical_covariates
            case "only_past":
                if dataset.future_covariates:
                    df.drop(columns=dataset.future_covariates, inplace=True)
                past_covs = dataset.past_covariates
                categorical_covs = [
                    cov for cov in dataset.categorical_covariates if cov not in dataset.future_covariates
                ]
            case "only_future":
                if dataset.past_covariates:
                    df.drop(columns=dataset.past_covariates, inplace=True)
                future_covs = dataset.future_covariates
                categorical_covs = [
                    cov for cov in dataset.categorical_covariates if cov not in dataset.past_covariates
                ]
            case "time":
                if all_covs:
                    df.drop(columns=all_covs, inplace=True)
                df, feature_names = self._add_time_features(df, dataset.granularity)
                future_covs = feature_names
                categorical_covs = feature_names
            case "lagged_target":
                if all_covs:
                    df.drop(columns=all_covs, inplace=True)
                lags: List[int] = get_lags_for_frequency(dataset.granularity, num_lags=7, num_default_lags=3)
                
                for target_col in sample.targets:
                    for lag in lags:
                        lagged_col_name: str = f"{target_col}_lag_{lag}"
                        if "ts_name" in dataset.df.columns and sample.ts_name != "_single_series_":
                            df[lagged_col_name] = df.groupby("ts_name")[target_col].shift(lag)
                        else:
                            df[lagged_col_name] = df[target_col].shift(lag)
                        past_covs.append(lagged_col_name)
                df.dropna(inplace=True) # Drop rows with NaN values created by shifting
            case "selected":
                selected_covs, df = self._cov_selection(sample, df, lagged=False)
                if df is None: # Error occurred in _cov_selection
                    return None
                
                # Keep only selected_covs, drop others
                cols_to_drop = [
                    cov for cov in all_covs if cov not in selected_covs and cov in df.columns
                ]
                if cols_to_drop:
                    df.drop(columns=cols_to_drop, inplace=True)
                
                past_covs = [cov for cov in selected_covs if cov in dataset.past_covariates]
                future_covs = [cov for cov in selected_covs if cov in dataset.future_covariates]
                categorical_covs = [cov for cov in selected_covs if cov in dataset.categorical_covariates]

            case "lagged_selected":
                selected_covs, df = self._cov_selection(sample, df, lagged=True)
                if df is None: # Error occurred in _cov_selection
                    return None
                # Keep only selected_covs, drop others
                actual_present_cov_cols = [col for col in df.columns if "_lag" in col]
                past_covs = [col for col in actual_present_cov_cols if any(pc in col for pc in dataset.past_covariates)]
                future_covs = [col for col in actual_present_cov_cols if any(fc in col for fc in dataset.future_covariates)]
                categorical_covs = [col for col in actual_present_cov_cols if any(cc in col for cc in dataset.categorical_covariates)]


        if self.max_ts_length:
            df = df.tail(self.max_ts_length)

        if df.empty:
            ds_logger.warning(
                f"DataFrame is empty after processing for sample ID {sample.id}. "
                "Returning None for ModelInput."
            )
            return None

        model_input: ModelInput = ModelInput(
            df=df,
            forecast_horizon=sample.fc_horizon,
            targets=sample.targets,
            past_covs=past_covs,
            future_covs=future_covs,
            categorical_covs=categorical_covs,
            date_col="date",
            frequency=dataset.granularity,
            quantile=self.quantile,
        )

        return model_input

    def get_forecast_actuals(self, sample_id: int) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Retrieves the training data and the actual target values for the forecast horizon
        of a specific sample, organized by time series ID (ts_id).

        This method loads the original dataset, identifies the time windows corresponding
        to the training period and the forecast period, and returns two dictionaries.
        Keys in both dictionaries are the ts_ids associated with the sample, and values
        are NumPy arrays of the target values for that ts_id.

        Parameters
        ----------
        sample_id : int
            ID of the sample for which to retrieve data.

        Returns
        -------
        Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]
            A tuple containing two dictionaries:
            - `training_data_dict`: Maps each ts_id (int) to a NumPy array of its
              target values for the training period. Shape: (training_length,).
            - `actuals_data_dict`: Maps each ts_id (int) to a NumPy array of its
              actual target values for the forecast period. Shape: (forecast_horizon,).
            Returns two empty dictionaries if errors occur or no data can be retrieved.

        Raises
        ------
        ValueError
            If the filtered DataFrame is empty for a specific time series name.
        """
        sample: Optional[Sample] = self.get_sample(sample_id)
        if sample is None:
            return {}, {}

        training_data_dict: Dict[int, np.ndarray] = {}
        actuals_data_dict: Dict[int, np.ndarray] = {}

        try:
            dataset: Dataset = Dataset(sample.dataset_name)
            full_df: pd.DataFrame = dataset.df.copy()

            # Filter for the specific time series name if the dataset contains multiple
            if "ts_name" in full_df.columns and sample.ts_name != "_single_series_":
                full_df_ts: pd.DataFrame = full_df[full_df["ts_name"] == sample.ts_name].copy()
                if full_df_ts.empty:
                    raise ValueError(
                        f"Filtered DataFrame is empty for ts_name '{sample.ts_name}' "
                        f"in dataset '{sample.dataset_name}'."
                    )
                full_df = full_df_ts
            elif "ts_name" in full_df.columns and sample.ts_name != "_single_series_":
                ds_logger.warning(f"ts_name '{sample.ts_name}' not found in dataset '{sample.dataset_name}'. Proceeding with full dataset if possible.")


            H: int = sample.fc_horizon
            w: int = sample.window
            total_len: int = len(full_df)

            # Calculate slice indices for actuals (forecast period)
            actuals_start_slice_idx: int = total_len - (w + 1) * H
            actuals_end_slice_idx: int = total_len - w * H

            # Calculate slice indices for training data
            training_end_slice_idx: int = actuals_start_slice_idx

            # Validate indices for actuals
            if actuals_start_slice_idx < 0:
                ds_logger.warning(
                    f"Actuals: Calculated start index {actuals_start_slice_idx} is negative "
                    f"for sample {sample_id} (len={total_len}, H={H}, w={w}). Clamping to 0."
                )
                actuals_start_slice_idx = 0
                if actuals_end_slice_idx <= 0:
                    ds_logger.error(
                        f"Actuals: Cannot retrieve actuals for sample {sample_id}. "
                        "Forecast window entirely precedes data start."
                    )
                    return {}, {}

            if actuals_end_slice_idx > total_len:
                ds_logger.warning(
                    f"Actuals: Calculated end index {actuals_end_slice_idx} exceeds series length "
                    f"{total_len} for sample {sample_id}. Clamping to end."
                )
                actuals_end_slice_idx = total_len
            elif actuals_end_slice_idx < actuals_start_slice_idx:
                ds_logger.error(
                    f"Actuals: Cannot retrieve actuals for sample {sample_id}. "
                    f"End index {actuals_end_slice_idx} is before start index {actuals_start_slice_idx}."
                )
                return {}, {}

            # Validate indices for training data
            training_period_df: pd.DataFrame
            if training_end_slice_idx < 0:
                ds_logger.warning(
                    f"Training: Calculated end index {training_end_slice_idx} is negative. "
                    "No training data can be extracted."
                )
                training_period_df = pd.DataFrame(columns=full_df.columns)
            else:
                training_period_df = full_df.iloc[:training_end_slice_idx]

            actuals_period_df: pd.DataFrame = full_df.iloc[actuals_start_slice_idx:actuals_end_slice_idx]

            if actuals_period_df.empty and H > 0:
                ds_logger.warning(
                    f"No actuals data found for sample {sample_id} in the calculated slice "
                    f"[{actuals_start_slice_idx}:{actuals_end_slice_idx}]."
                )

            # Iterate through the ts_ids associated with the sample to get target values
            for ts_id in sample.ts_ids:
                try:
                    ts_id_info: TSInfo = TSInfo(ts_ids=[ts_id])
                    if not ts_id_info.target_names:
                        ds_logger.warning(f"Could not find target name for ts_id {ts_id} in TSInfo. Skipping.")
                        continue
                    target_name: str = ts_id_info.target_names[0]

                    # Process Training Data
                    if target_name in training_period_df.columns and not training_period_df.empty:
                        training_values: np.ndarray = training_period_df[target_name].to_numpy()
                        training_data_dict[ts_id] = training_values
                    elif not training_period_df.empty:
                        ds_logger.warning(
                            f"Training: Target column '{target_name}' for ts_id {ts_id} "
                            f"not found in training data for sample {sample_id}."
                        )

                    # Process Actuals Data
                    if target_name in actuals_period_df.columns and not actuals_period_df.empty:
                        actual_values: np.ndarray = actuals_period_df[target_name].to_numpy()
                        actuals_data_dict[ts_id] = actual_values
                    elif not actuals_period_df.empty:
                        ds_logger.warning(
                            f"Actuals: Target column '{target_name}' for ts_id {ts_id} "
                            f"not found in actuals data for sample {sample_id}."
                        )

                except IndexError:
                    ds_logger.warning(f"Error retrieving information or target name for ts_id {ts_id}. Skipping.")
                except Exception as e_inner:
                    ds_logger.warning(
                        f"Error processing ts_id {ts_id} within sample {sample_id}: {e_inner}"
                    )

            if not training_data_dict and not actuals_data_dict and total_len > 0:
                ds_logger.warning(
                    f"Both training and actuals data dictionaries are empty for sample {sample_id}. "
                    "This might indicate an issue with target names or data slicing."
                )

            return training_data_dict, actuals_data_dict

        except (FileNotFoundError, ValueError, KeyError, IndexError) as e:
            ds_logger.error(
                f"Error getting training/forecast actuals dict for sample ID {sample_id} "
                f"(Dataset: {sample.dataset_name}, TS: {sample.ts_name}): {e}"
            )
            return {}, {}
        except Exception as e:
            ds_logger.error(
                f"Unexpected error getting training/forecast actuals dict for sample ID {sample_id}: {e}",
                exc_info=True,
            )
            return {}, {}

    def get_dimension_and_length(self, sample_id: int, mode: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Retrieves the dimension (number of features) and length (number of time steps)
        of the processed time series data for a specific sample and mode.

        Parameters
        ----------
        sample_id : int
            ID of the sample to retrieve.
        mode : str
            The covariate handling mode to apply when processing the sample.

        Returns
        -------
        Tuple[Optional[int], Optional[int]]
            A tuple containing:
            - The dimension of the time series data (number of columns in `df`).
            - The length of the time series data (number of rows in `df`),
              adjusted for future covariates.
            Returns (None, None) if `ModelInput` cannot be generated.
        """
        model_input: Optional[ModelInput] = self.get_model_input(sample_id=sample_id, mode=mode)
        if model_input is None:
            ds_logger.error(
                f"Model input is None for sample ID {sample_id}. Cannot determine dimension and length."
            )
            return None, None

        length: int = len(model_input.df)
        if model_input.future_covs:  # Check if the list is not empty
            length -= model_input.forecast_horizon

        # Dimension is the number of target series + past covariates + future covariates
        dimension: int = len(model_input.targets) + len(model_input.past_covs) + len(model_input.future_covs)
        return dimension, length