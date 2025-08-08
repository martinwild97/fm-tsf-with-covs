import ast
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, rankdata

from src.data.data_loader import COV_MODES, DATA_DIR, DataLoader
from src.evaluation.metrics import Metrics

# --- Configuration Paths ---
FORECAST_DIR: Path = DATA_DIR / "forecasts"
PLOTS_DIR: Path = DATA_DIR / "plots"
DEFAULT_METRICS_PATH: Path = DATA_DIR / "metrics.csv"
DEFAULT_METRICS_AGG_PATH: Path = DATA_DIR / "metrics_aggregated.csv"
LOG_DIR: Path = Path(__file__).resolve().parent.parent.parent / "logs"
DEFAULT_LOG_PATH: Path = LOG_DIR / "evaluation.log"
DEFAULT_SAMPLE_PATH: Path = DATA_DIR / "sample_info.csv"
DEFAULT_DIM_LENGTH_PATH: Path = DATA_DIR / "dim_length.csv"

# --- Forecast File Names ---
DEFAULT_DATA_FILENAME: str = "data.csv"
DEFAULT_META_FILENAME: str = "meta.csv"

# --- Model Definitions ---
# Full list of all models expected in the raw data
ALL_MODELS: List[str] = [
    "AutoARIMA",
    "AutoETS",
    "AutoTheta",
    "gbm",
    "chronos_base",
    "chronos_gbm",
    "timegpt",
    "timegpt_finetuned",
    "timegpt_finetuned2",
    "ttm",
    "ttm_finetuned",
    "timesfm_v2",
    "moirai_base",
    "moirai_moe",
    "TiDE",
    "NBEATSx",  # "TFT"
]

# User-friendly names for models, used in plots and aggregated results
PRETTY_MODELS: List[str] = [
    "(S)ARIMAx",
    "ETS",
    "Theta",
    "LightGBM",
    "Chronos + LR",
    "Chronos + GBM",
    "TimeGPT",
    "TimeGPT Finetuned",
    "TTM",
    "TTM Finetuned",
    "TimesFM",
    "MOIRAI",
    "MOIRAI MoE",
    "TiDE",
    "NBEATSx",
]

# User-friendly names for covariate modes
PRETTY_MODES: List[str] = [
    "All",
    "No Exogenous",
    "Noise",
    "Only Past",
    "Only Future",
    "Time",
    "Lagged Target",
    "Selected",
    "Selected with Lags",
]

# Default configuration for checking forecast availability or general analysis
# Each tuple represents (list_of_models, list_of_modes) to group runs.
DEFAULT_CONFIG: List[Tuple[List[str], List[str]]] = [
    (PRETTY_MODELS, ["No Exogenous"]),
    (
        [model for model in PRETTY_MODELS if model not in ["ETS", "Theta"]],
        ["All"],
    ),
    (
        [
            model
            for model in PRETTY_MODELS
            if model
            not in [
                "(S)ARIMAx",
                "ETS",
                "Theta",
                "LightGBM",
                "TiDE",
                "NBEATSx",
            ]
        ],
        [
            "Noise",
            "Only Past",
            "Only Future",
            "Time",
            "Lagged Target",
            "Selected",
            "Selected with Lags",
        ],
    ),
]

# Categorization of models for plotting/analysis purposes
MODEL_CATEGORIES: Dict[str, str] = {
    "(S)ARIMAx": "Statistical",
    "ETS": "Statistical",
    "Theta": "Statistical",
    "LightGBM": "Machine Learning",
    "Chronos + LR": "Foundation Model",
    "Chronos + GBM": "Foundation Model",
    "TimeGPT": "Foundation Model",
    "TimeGPT Finetuned": "Foundation Model",
    "TimeGPT Finetuned 2": "Foundation Model",
    "TTM": "Foundation Model",
    "TTM Finetuned": "Foundation Model",
    "TimesFM": "Foundation Model",
    "MOIRAI": "Foundation Model",
    "MOIRAI MoE": "Foundation Model",
    "TFT": "Deep Learning",
    "TiDE": "Deep Learning",
    "NBEATSx": "Deep Learning",
}

# Color mapping for model categories in plots
CATEGORY_COLORS: Dict[str, str] = {
    "Foundation Model": "#1f77b4",  # Matplotlib default blue
    "Statistical": "#ff7f0e",  # Matplotlib default orange
    "Deep Learning": "#2ca02c",  # Matplotlib default green
    "Machine Learning": "#9467bd",  # Matplotlib default purple
}

# Predefined lists of ts_ids that have certain covariate types
TS_IDS_WITH_FUTURE_COVS: List[int] = [
    2,
    3,
    4,
    5,
    6,
    1083,
    1084,
    1126,
    1128,
    1129,
    1133,
    1139,
    1140,
    1143,
    1146,
    1150,
    1154,
    1155,
    1159,
    1162,
    1166,
    1176,
    1179,
    1180,
    1190,
    1193,
    1195,
    1205,
    1209,
    1211,
    1216,
    1222,
    1230,
    1234,
    1241,
    1248,
    1253,
    1258,
    1259,
    1271,
    1292,
    1293,
    1303,
    1319,
    1321,
    1322,
    1332,
    1399,
]
TS_IDS_WITH_PAST_COVS: List[int] = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    12,
    14,
    21,
    24,
    35,
    36,
    39,
    42,
    44,
    59,
    61,
    75,
    77,
    83,
    85,
    90,
    91,
    100,
    112,
    114,
    116,
    121,
    123,
    128,
    129,
    130,
    131,
    132,
    133,
    141,
    143,
    147,
    148,
    149,
    150,
    151,
    556,
    557,
    600,
    601,
    602,
    605,
    606,
    611,
    615,
    619,
    628,
    638,
    641,
    642,
    643,
    656,
    794,
    854,
    968,
    980,
    1083,
    1084,
    1085,
    1086,
    1087,
    1088,
    1089,
    1105,
    1107,
    1108,
    1109,
    1110,
]


# --- Directory Setup ---
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# --- Logger Configuration ---
def setup_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """
    Sets up and configures a logger with both console and file handlers.

    Parameters
    ----------
    log_path : Path, default=DEFAULT_LOG_PATH
        The full path to the log file. Parent directories will be created if they don't exist.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    logger_instance = logging.getLogger("Evaluation")
    # Clear existing handlers to prevent duplicate log entries if called multiple times
    if logger_instance.hasHandlers():
        logger_instance.handlers.clear()

    logger_instance.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger_instance.addHandler(console_handler)

    # File handler
    log_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger_instance.addHandler(file_handler)

    return logger_instance


logger: logging.Logger = setup_logger()


class ForecastEvaluator:
    """
    Evaluates time series forecasts by comparing them against actual values
    using various metrics and optionally generates plots. It handles loading
    forecast data, metadata, and actuals from a DataLoader.
    """

    def __init__(
        self,
        dataloader: Optional[DataLoader] = None,
        data_file_path: Path = FORECAST_DIR / DEFAULT_DATA_FILENAME,
        meta_file_path: Path = FORECAST_DIR / DEFAULT_META_FILENAME,
        metrics_output_path: Path = DEFAULT_METRICS_PATH,
        metrics_aggregated_output_path: Path = DEFAULT_METRICS_AGG_PATH,
        model_categories: Optional[Dict[str, str]] = MODEL_CATEGORIES,
        dim_length_output_path: Path = DEFAULT_DIM_LENGTH_PATH,
    ) -> None:
        """
        Initializes the ForecastEvaluator.

        Parameters
        ----------
        dataloader : Optional[DataLoader], default=None
            An initialized DataLoader instance to retrieve actual values and training data.
            If None, a default DataLoader instance will be created.
        data_file_path : Path, default=FORECAST_DIR / "data.csv"
            Path to the CSV file containing detailed forecast data (point forecasts, intervals).
        meta_file_path : Path, default=FORECAST_DIR / "meta.csv"
            Path to the CSV file containing metadata about the forecasts (model, mode, sample_id, success, etc.).
        metrics_output_path : Path, default=DEFAULT_METRICS_PATH
            Path to save the detailed, per-time-series-per-sample metrics results.
        metrics_aggregated_output_path : Path, default=DEFAULT_METRICS_AGG_PATH
            Path to save the aggregated metrics (e.g., mean MASE across all time series).
        model_categories : Optional[Dict[str, str]], default=MODEL_CATEGORIES
            Dictionary mapping user-friendly model names to their categories (e.g., 'Statistical', 'Foundation Model').
        dim_length_output_path : Path, default=DEFAULT_DIM_LENGTH_PATH
            Path to a CSV file storing time series dimensions and lengths for each sample/mode combination.
        """
        self.dl: DataLoader = dataloader if dataloader is not None else DataLoader()
        self.metrics_output_path: Path = metrics_output_path
        self.metrics_aggregated_output_path: Path = metrics_aggregated_output_path
        self.model_categories: Optional[Dict[str, str]] = model_categories
        self.dim_length_output_path: Path = dim_length_output_path
        self.category_colors: Dict[str, str] = CATEGORY_COLORS
        self.metrics: Metrics = Metrics()
        self.ts_ids_with_past_covs: List[int] = TS_IDS_WITH_PAST_COVS
        self.ts_ids_with_future_covs: List[int] = TS_IDS_WITH_FUTURE_COVS
        self.sample_ids: List[int] = [s.id for s in self.dl.samples]

        # Mappings for internal (raw) vs. pretty (display) names for modes and models
        self.mode_map: Dict[str, str] = {
            "all": "All",
            "lagged_selected": "Selected with Lags",
            "lagged_target": "Lagged Target",
            "no": "No Exogenous",
            "noise": "Noise",
            "only_future": "Only Future",
            "only_past": "Only Past",
            "selected": "Selected",
            "time": "Time",
        }
        self.mode_reverse_map: Dict[str, str] = {v: k for k, v in self.mode_map.items()}

        self.model_map: Dict[str, str] = {
            "chronos_base": "Chronos + LR",
            "chronos_gbm": "Chronos + GBM",
            "gbm": "LightGBM",
            "moirai_base": "MOIRAI",
            "moirai_moe": "MOIRAI MoE",
            "timegpt": "TimeGPT",
            "timegpt_finetuned": "TimeGPT Finetuned",
            "timesfm_v2": "TimesFM",
            "ttm": "TTM",
            "ttm_finetuned": "TTM Finetuned",
            # Other models, ensure all `ALL_MODELS` are mapped
            "AutoARIMA": "(S)ARIMAx",
            "AutoETS": "ETS",
            "AutoTheta": "Theta",
            "moirai_batch2": "MOIRAI2",
            "timegpt_finetuned2": "TimeGPT Finetuned 2",
            "TiDE": "TiDE",
            "NBEATSx": "NBEATSx",
        }
        self.model_reverse_map: Dict[str, str] = {v: k for k, v in self.model_map.items()}

        # Load dataframes during initialization
        self.df_data: Optional[pd.DataFrame] = self._load_csv_file(
            data_file_path,
            required_cols=["forecast_id", "ts_id", "point_forecast", "lower_bound", "upper_bound"],
        )
        self.df_meta: Optional[pd.DataFrame] = self._load_csv_file(
            meta_file_path,
            required_cols=["model", "mode", "sample_id", "forecast_successful", "forecast_time_seconds", "forecast_ids"],
        )

        # Process metadata DataFrame
        if self.df_meta is not None:
            self.df_meta["forecast_successful"] = (
                self.df_meta["forecast_successful"].astype(str).str.lower() == "true"
            )
            self.df_meta["forecast_ids_list"] = self.df_meta["forecast_ids"].apply(self._safe_parse_ids)
            # Apply renaming to pretty names immediately after loading df_meta
            self.df_meta["model"] = self.df_meta["model"].map(self.model_map).fillna(self.df_meta["model"])
            self.df_meta["mode"] = self.df_meta["mode"].map(self.mode_map).fillna(self.df_meta["mode"])

        # Load or generate dimension and length data
        if self.dim_length_output_path.exists():
            try:
                self.df_dim_length: Optional[pd.DataFrame] = pd.read_csv(self.dim_length_output_path)
            except Exception as e:
                logger.warning(
                    f"Could not read dimension/length file {self.dim_length_output_path}: {e}"
                )
                self.df_dim_length = None
        else:
            self.df_dim_length = self._generate_dim_length_dataframe()

        if self.df_dim_length is not None:
            # Apply renaming to pretty names for df_dim_length
            self.df_dim_length["mode"] = (
                self.df_dim_length["mode"].map(self.mode_map).fillna(self.df_dim_length["mode"])
            )

    def _load_csv_file(self, file_path: Path, required_cols: List[str]) -> Optional[pd.DataFrame]:
        """
        Loads a single CSV file into a pandas DataFrame, handling potential errors and column checks.

        Parameters
        ----------
        file_path : Path
            The full path to the CSV file to load.
        required_cols : List[str]
            A list of column names that must be present in the loaded DataFrame.

        Returns
        -------
        Optional[pd.DataFrame]
            The loaded DataFrame if successful and all required columns are present,
            otherwise None.
        """
        if not file_path.exists():
            logger.error(f"Data file not found: {file_path}")
            return None
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                logger.warning(f"Loaded DataFrame from {file_path} is empty.")
                return pd.DataFrame(columns=required_cols)

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"File {file_path} is missing required columns: {missing_cols}")
                return None

            logger.info(f"Successfully loaded data from {file_path} ({len(df)} rows).")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return None

    def _safe_parse_ids(self, val: Any) -> List[int]:
        """
        Safely parses a string representation of a list or tuple of IDs into a list of integers.
        Handles various input formats and potential errors during parsing.

        Parameters
        ----------
        val : Any
            The value to parse, expected to be a string representation of a list or tuple.

        Returns
        -------
        List[int]
            A list of integers parsed from the input. Returns an empty list on failure.
        """
        try:
            # Ensure the input is treated as a string before literal_eval
            ids_parsed = ast.literal_eval(str(val))
            if isinstance(ids_parsed, (list, tuple)):
                # Convert each element to int, handling cases where they might be float or string
                return [int(i) for i in ids_parsed if isinstance(i, (int, float, str))]
            return []
        except (ValueError, SyntaxError, TypeError):
            logger.debug(f"Could not parse IDs from value: {val}. Returning empty list.")
            return []

    def _get_forecast_arrays(self, forecast_ids: List[int]) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Retrieves and structures forecast data (point, lower, upper bounds)
        for a given set of `forecast_id`s.

        Parameters
        ----------
        forecast_ids : List[int]
            A list of `forecast_id`s to retrieve data for.

        Returns
        -------
        Dict[int, Dict[str, np.ndarray]]
            A dictionary where keys are `ts_id`s and values are dictionaries
            containing 'point', 'lower', and 'upper' forecast arrays.
            Returns an empty dictionary if `df_data` is not loaded or no IDs are provided/found.
        """
        if self.df_data is None or not forecast_ids:
            return {}

        # Ensure forecast_ids are integers for proper filtering
        forecast_ids_int: List[int] = [int(fid) for fid in forecast_ids]
        filtered_df: pd.DataFrame = self.df_data[
            self.df_data["forecast_id"].isin(forecast_ids_int)
        ].copy()

        if filtered_df.empty:
            logger.warning(f"No forecast data found for provided forecast_ids: {forecast_ids}.")
            return {}

        # Sort values to ensure correct order for time series steps
        filtered_df.sort_values(by=["ts_id", "step"], inplace=True)
        forecasts_by_ts: Dict[int, Dict[str, np.ndarray]] = {}

        for ts_id, group in filtered_df.groupby("ts_id"):
            try:
                ts_id_int: int = int(ts_id)
            except ValueError:
                logger.warning(f"Non-integer ts_id '{ts_id}' encountered. Skipping.")
                continue

            point_fc: np.ndarray = group["point_forecast"].to_numpy(dtype=float)
            # Check for column existence and non-NaN values before converting to numpy
            lower_fc: Optional[np.ndarray] = (
                group["lower_bound"].to_numpy(dtype=float)
                if "lower_bound" in group.columns and group["lower_bound"].notna().any()
                else None
            )
            upper_fc: Optional[np.ndarray] = (
                group["upper_bound"].to_numpy(dtype=float)
                if "upper_bound" in group.columns and group["upper_bound"].notna().any()
                else None
            )

            forecasts_by_ts[ts_id_int] = {
                "point": point_fc,
                "lower": lower_fc,
                "upper": upper_fc,
            }
        return forecasts_by_ts

    def _plot_save_and_show(
        self,
        fig: plt.Figure,
        save_path: Path,
        filename: str,
        show_plot: bool = True,
        save_plot: bool = False,
    ) -> None:
        """
        Helper function to save and display a matplotlib plot.

        Parameters
        ----------
        fig : plt.Figure
            The matplotlib Figure object to save/show.
        save_path : Path
            The directory where the plot should be saved.
        filename : str
            The name of the file to save the plot as (e.g., "my_plot.png").
        show_plot : bool, default=True
            If True, displays the plot.
        save_plot : bool, default=False
            If True, saves the plot to the specified path and filename.
        """
        if save_plot:
            try:
                save_path.mkdir(parents=True, exist_ok=True)  # Ensure save directory exists
                fig.savefig(save_path / filename, bbox_inches="tight")
                logger.info(f"Plot saved to {save_path / filename}")
            except Exception as e:
                logger.error(f"Error saving plot to {save_path / filename}: {e}")

        if show_plot:
            plt.show()
        plt.close(fig)  # Close the figure to free up memory

    def _generate_dim_length_dataframe(self) -> pd.DataFrame:
        """
        Generates a DataFrame containing the dimension (number of features)
        and length (number of time steps) for each sample and mode combination.
        Saves this DataFrame to `self.dim_length_output_path`.

        Returns
        -------
        pd.DataFrame
            A DataFrame with 'sample_id', 'mode', 'dimension', and 'length' columns.
            Returns an empty DataFrame if no samples are found.
        """
        dim_length_data: List[Dict[str, Union[int, str, float]]] = []

        for s_id in self.sample_ids:
            # Iterate through raw COV_MODES as DataLoader expects them
            for mode_raw in COV_MODES:
                try:
                    dim, length = self.dl.get_dimension_and_length(s_id, mode_raw)
                    dim_length_data.append(
                        {"sample_id": s_id, "mode": mode_raw, "dimension": dim, "length": length}
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not get dimension/length for sample_id {s_id}, mode {mode_raw}: {e}. "
                        "Appending NaN values."
                    )
                    dim_length_data.append(
                        {"sample_id": s_id, "mode": mode_raw, "dimension": np.nan, "length": np.nan}
                    )

        dim_length_df: pd.DataFrame = pd.DataFrame(dim_length_data)
        try:
            dim_length_df.to_csv(self.dim_length_output_path, index=False)
            logger.info(f"Dimension/length data saved to {self.dim_length_output_path}")
        except Exception as e:
            logger.error(f"Error saving dimension/length data to CSV: {e}")

        return dim_length_df

    def get_timing_dataframe(
        self,
        models: Optional[List[str]] = None,
        modes: Optional[List[str]] = None,
        sample_ids: Optional[List[int]] = None,
        require_all_samples: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieves timing data (forecast_time_seconds) for specified models, modes, and samples.

        Parameters
        ----------
        models : Optional[List[str]], default=PRETTY_MODELS
            A list of user-friendly model names to filter by. If None, uses all `PRETTY_MODELS`.
        modes : Optional[List[str]], default=["All"]
            A list of user-friendly mode names to filter by. If None, uses ["All"].
        sample_ids : Optional[List[int]], default=None
            A list of sample IDs to filter by. If None, considers all sample IDs available in DataLoader.
        require_all_samples : bool, default=False
            If True, only includes `sample_id`s that have timing data for ALL
            specified model-mode combinations.

        Returns
        -------
        Optional[pd.DataFrame]
            A DataFrame containing 'model', 'mode', 'sample_id', 'forecast_time_seconds',
            'dimension', and 'length' columns, or None if metadata is not loaded or no data found.
        """
        if self.df_meta is None:
            logger.error("Metadata not loaded. Cannot retrieve timing data.")
            return None

        # Set default values if not provided
        models = models if models is not None else PRETTY_MODELS
        modes = modes if modes is not None else ["All"]

        filtered_df: pd.DataFrame = self.df_meta[
            (self.df_meta["model"].isin(models)) & (self.df_meta["mode"].isin(modes))
        ].copy()

        if sample_ids is not None:
            filtered_df = filtered_df[filtered_df["sample_id"].isin(sample_ids)]

        # Filter to only include sample_ids known to the DataLoader
        filtered_df = filtered_df[filtered_df["sample_id"].isin(self.sample_ids)]

        # Drop rows where forecast_time_seconds is NaN
        filtered_df = filtered_df[filtered_df["forecast_time_seconds"].notna()]

        if require_all_samples:
            # Determine the expected number of combinations for each sample_id
            expected_count_per_sample = len(models) * len(modes)
            # Count how many combinations each sample_id actually has
            actual_counts_per_sample = filtered_df["sample_id"].value_counts()
            # Find sample_ids that have the expected count
            complete_sample_ids = actual_counts_per_sample[
                actual_counts_per_sample == expected_count_per_sample
            ].index.tolist()
            filtered_df = filtered_df[filtered_df["sample_id"].isin(complete_sample_ids)]

        # Select only relevant columns for the output and merge with dimension/length
        filtered_df = filtered_df[["model", "mode", "sample_id", "forecast_time_seconds"]].copy()

        if filtered_df.empty:
            logger.warning(
                f"No timing data found for models {models}, modes {modes}, and sample IDs {sample_ids} "
                "after filtering and cleaning."
            )
            return None
        else:
            logger.info(
                f"Retrieved timing data for {len(filtered_df)} entries with models {models}, "
                f"modes {modes}, and specified sample IDs."
            )

        # Merge with dimension and length information
        if self.df_dim_length is not None:
            final_df: pd.DataFrame = filtered_df.merge(
                self.df_dim_length, on=["sample_id", "mode"], how="left"
            )
            # Log how many unique sample_ids are included after merging
            included_sample_ids_count = final_df["sample_id"].nunique()
            logger.info(
                f"Included {included_sample_ids_count} sample IDs out of {self.dl._num_samples} "
                "total sample IDs in timing results."
            )
        else:
            logger.warning("Dimension/length data not available. Timing data will not include these columns.")
            final_df = filtered_df # Use filtered_df as final if dim/length is missing

        return final_df

    def check_available_forecasts(
        self, config: Optional[List[Tuple[List[str], List[str]]]] = DEFAULT_CONFIG
    ) -> None:
        """
        Checks and logs the availability of forecasts based on a provided configuration.
        It reports on successful, failed, and missing forecasts for each model-mode pair.

        Parameters
        ----------
        config : Optional[List[Tuple[List[str], List[str]]]], default=DEFAULT_CONFIG
            A list of tuples, where each tuple contains (list_of_models, list_of_modes).
            This defines the combinations to check.
        """
        if self.df_meta is None or self.df_data is None:
            logger.error("Metadata or forecast data not loaded. Cannot check available forecasts.")
            return

        # Filter meta data to only include sample_ids known to DataLoader
        df_meta_filtered: pd.DataFrame = self.df_meta[self.df_meta["sample_id"].isin(self.sample_ids)].copy()

        # Get all unique forecast_ids present in the actual forecast data file
        df_data_ids: set[int] = set(self.df_data["forecast_id"].unique())

        # Identify explicitly failed forecasts
        df_failed: pd.DataFrame = df_meta_filtered.loc[
            ~df_meta_filtered["forecast_successful"], ["model", "mode", "sample_id"]
        ]

        # Consider only successful forecasts that also have associated forecast_ids
        df_success: pd.DataFrame = df_meta_filtered.loc[
            self.df_meta["forecast_successful"] & self.df_meta["forecast_ids_list"].map(len).gt(0)
        ].copy()
        
        # Explode the list of forecast_ids to check individual ID presence in df_data
        df_success_exploded: pd.DataFrame = df_success.explode("forecast_ids_list")
        df_success_exploded["in_data"] = df_success_exploded["forecast_ids_list"].isin(df_data_ids)

        # Aggregate to check if all forecast IDs for a given (model, mode, sample_id) are present in data
        agg_presence = (
            df_success_exploded.groupby(["model", "mode", "sample_id"], as_index=False)
            .agg(all_ids_present=("in_data", "all"))
        )

        results: List[Dict[str, Any]] = []
        # `missing_counter` tracks which sample_ids are frequently missing across combinations
        missing_counter: Dict[int, int] = {}
        total_runs_checked: int = 0

        for models_subset, modes_subset in config:
            for model in models_subset:
                for mode in modes_subset:
                    total_runs_checked += 1
                    
                    # Get sample IDs where the forecast explicitly failed
                    failed_ids: set[int] = set(
                        df_failed.loc[
                            (df_failed["model"] == model) & (df_failed["mode"] == mode),
                            "sample_id",
                        ].astype(int)
                    )

                    # Get sample IDs where all associated forecast_ids were successfully found in df_data
                    current_success_agg = agg_presence.loc[
                        (agg_presence["model"] == model) & (agg_presence["mode"] == mode)
                    ]
                    available_ids: set[int] = set(
                        current_success_agg.loc[current_success_agg["all_ids_present"], "sample_id"].astype(int)
                    )

                    # Missing IDs are those that are not failed and not available (i.e., meta entry exists but data is incomplete)
                    # This implies there was a meta entry for this (model, mode, sample_id) but `all_ids_present` is False
                    missing_ids_from_data_check: set[int] = set(
                        current_success_agg.loc[~current_success_agg["all_ids_present"], "sample_id"].astype(int)
                    )

                    # Calculate truly missing from expected set of sample_ids
                    total_expected_samples: int = len(self.sample_ids)
                    
                    # Total number of actual entries for this model/mode combination in df_meta
                    total_meta_entries = df_meta_filtered[
                        (df_meta_filtered['model'] == model) &
                        (df_meta_filtered['mode'] == mode)
                    ]['sample_id'].nunique()
                    
                    # Number of samples that were expected but have no meta entry at all
                    # This implicitly means they are also 'missing' or 'failed'
                    implicitly_missing_ids = set(self.sample_ids) - (available_ids | failed_ids | missing_ids_from_data_check)


                    current_available_count = len(available_ids)
                    current_failed_count = len(failed_ids)
                    # Combine missing due to incomplete data and implicitly missing
                    current_missing_count = len(missing_ids_from_data_check | implicitly_missing_ids)

                    results.append(
                        {
                            "model": model,
                            "mode": mode,
                            "available_pct": (current_available_count / total_expected_samples) * 100,
                            "failed_pct": (current_failed_count / total_expected_samples) * 100,
                            "missing_pct": (current_missing_count / total_expected_samples) * 100,
                            "available_ids": sorted(list(available_ids)),
                            "failed_ids": sorted(list(failed_ids)),
                            "missing_ids": sorted(list(missing_ids_from_data_check | implicitly_missing_ids)),
                        }
                    )

                    # Update missing counter for overall analysis
                    for sid in (missing_ids_from_data_check | implicitly_missing_ids):
                        missing_counter[sid] = missing_counter.get(sid, 0) + 1

        for res in results:
            logger.info(
                f"Model: {res['model']} | Mode: {res['mode']} | Available: {res['available_pct']:.2f}% | "
                f"Failed: {res['failed_pct']:.2f}% | Missing: {res['missing_pct']:.2f}%"
            )

        df_results_summary: pd.DataFrame = pd.DataFrame(results)
        logger.info(
            f"Overall Average Available: {df_results_summary['available_pct'].mean():.2f}% | "
            f"Overall Average Failed: {df_results_summary['failed_pct'].mean():.2f}% | "
            f"Overall Average Missing: {df_results_summary['missing_pct'].mean():.2f}%"
        )
        if missing_counter:
            most_frequent_missing = sorted(missing_counter.items(), key=lambda item: item[1], reverse=True)[:5]
            logger.info(f"Top 5 most frequently missing sample IDs: {most_frequent_missing}")

    def calculate_all_metrics(self) -> None:
        """
        Calculates all relevant evaluation metrics for all successful forecasts
        found in the loaded metadata and appends them to `self.metrics_output_path`.
        It skips metrics that have already been computed for a given model-mode-sample_id combination.
        """
        if self.df_meta is not None:
            pass


        if self.df_meta is None or self.df_data is None or self.dl is None:
            logger.error(
                "Cannot evaluate: Forecast metadata, data, or DataLoader is not loaded/initialized."
            )
            return

        existing_keys: set[Tuple[str, str, int]] = set()
        if self.metrics_output_path.exists():
            try:
                df_existing: pd.DataFrame = pd.read_csv(self.metrics_output_path)
                # Ensure existing_keys are formed using the same "pretty" names if applicable
                df_existing['model'] = df_existing['model'].map(self.model_map).fillna(df_existing['model'])
                df_existing['mode'] = df_existing['mode'].map(self.mode_map).fillna(df_existing['mode'])
                existing_keys = set(
                    zip(df_existing["model"], df_existing["mode"], df_existing["sample_id"])
                )
                logger.info(f"Found {len(existing_keys)} existing metric entries. Skipping these.")
            except Exception as e:
                logger.warning(
                    f"Could not read existing metrics file {self.metrics_output_path}: {e}. "
                    "Proceeding as if no existing metrics."
                )
                # If error, treat as if no existing data to avoid skipping valid new data.

        # Filter for successful forecasts with actual IDs that are not yet computed
        meta_to_evaluate: pd.DataFrame = self.df_meta.loc[
            self.df_meta["forecast_successful"]
            & self.df_meta["forecast_ids_list"].map(len).gt(0)
        ].copy()

        # Filter out already computed keys
        meta_to_evaluate["key"] = list(
            zip(meta_to_evaluate["model"], meta_to_evaluate["mode"], meta_to_evaluate["sample_id"])
        )
        meta_to_evaluate = meta_to_evaluate[~meta_to_evaluate["key"].isin(existing_keys)]
        meta_to_evaluate = meta_to_evaluate.drop(columns="key")

        if meta_to_evaluate.empty:
            logger.info("All metrics already computed or no new successful forecasts to evaluate.")
            return

        results: List[Dict[str, Any]] = []
        errors: int = 0
        total_forecasts_to_process: int = len(meta_to_evaluate)
        logger.info(f"Starting metric calculation for {total_forecasts_to_process} new forecasts.")

        for idx, row in meta_to_evaluate.iterrows():
            model: str = row["model"]
            mode: str = row["mode"]
            sample_id: int = int(row["sample_id"])
            ids: List[int] = row["forecast_ids_list"]

            try:
                train_data_by_ts, actuals_by_ts = self.dl.get_forecast_actuals(sample_id)
                forecasts_by_ts = self._get_forecast_arrays(ids)

                if not actuals_by_ts or not forecasts_by_ts:
                    logger.warning(
                        f"Skipping sample {sample_id} (Model: {model}, Mode: {mode}): "
                        "No actuals or forecasts found for any ts_id in this sample."
                    )
                    errors += 1 # Count this as an error for the sample
                    continue

                for ts_id, y_true in actuals_by_ts.items():
                    fore_data: Optional[Dict[str, np.ndarray]] = forecasts_by_ts.get(ts_id)
                    y_train: Optional[np.ndarray] = train_data_by_ts.get(ts_id)

                    if fore_data is None:
                        logger.warning(
                            f"Forecast data missing for ts_id {ts_id} in sample {sample_id} "
                            f"(Model: {model}, Mode: {mode}). Skipping this TS."
                        )
                        continue

                    y_pred: np.ndarray = fore_data["point"]
                    y_lower: Optional[np.ndarray] = fore_data.get("lower")
                    y_upper: Optional[np.ndarray] = fore_data.get("upper")

                    if len(y_true) != len(y_pred):
                        logger.warning(
                            f"Forecast length mismatch for ts_id {ts_id} in sample {sample_id} "
                            f"(Model: {model}, Mode: {mode}). Actuals length: {len(y_true)}, "
                            f"Forecast length: {len(y_pred)}. Skipping this TS."
                        )
                        continue
                    if y_train is None or len(y_train) == 0:
                        logger.warning(
                            f"Training data missing or empty for ts_id {ts_id} in sample {sample_id} "
                            f"(Model: {model}, Mode: {mode}). MASE/RMSSE/MSIS might be impacted. Skipping this TS."
                        )
                        continue # MASE/RMSSE/MSIS require training data, so skip if missing

                    current_metrics: Dict[str, Any] = {
                        "model": model,
                        "mode": mode,
                        "sample_id": sample_id,
                        "ts_id": ts_id,
                        "forecast_horizon": len(y_true),
                    }
                    m = self.metrics # Shorthand for Metrics class methods

                    try:
                        current_metrics["mase"] = m.mase(y_true, y_pred, y_train)
                        current_metrics["rmsse"] = m.rmsse(y_true, y_pred, y_train)
                        current_metrics["cfe"] = m.cfe(y_true, y_pred)
                        current_metrics["pis"] = m.pis(y_true, y_pred)

                        if y_lower is not None and y_upper is not None and len(y_lower) == len(y_true):
                            quantile_level_val = int(self.dl.quantile * 100)
                            current_metrics["msis"] = m.msis(
                                y_true, y_lower, y_upper, y_train, quantile_level=quantile_level_val
                            )
                            current_metrics["wql"] = m.wql(
                                y_true, y_pred, y_lower, y_upper, quantile_level=quantile_level_val
                            )
                        else:
                            current_metrics["msis"] = np.nan
                            current_metrics["wql"] = np.nan

                        results.append(current_metrics)

                    except Exception as e:
                        logger.error(
                            f"Error calculating metrics for ts_id {ts_id} in sample {sample_id} "
                            f"(Model: {model}, Mode: {mode}): {e}"
                        )
                        errors += 1
            except Exception as e:
                logger.error(f"Sample {sample_id} processing failed (Model: {model}, Mode: {mode}): {e}")
                errors += 1

        if results:
            df_new_metrics: pd.DataFrame = pd.DataFrame(results)
            # Ensure the output CSV uses the user-friendly names that are in `results`
            # and were populated from `meta_to_evaluate` (which uses pretty names).
            file_exists: bool = self.metrics_output_path.exists()
            df_new_metrics.to_csv(self.metrics_output_path, mode="a", header=not file_exists, index=False)
            logger.info(
                f"Appended {len(results)} new metrics rows to '{self.metrics_output_path}'."
            )
        logger.info(f"Metric calculation complete: {len(results)} total new metrics rows calculated, {errors} errors during processing.")

    def get_aggregated_metrics_dataframe(
        self,
        metric: str,
        modes: Union[str, List[str]],
        models: Union[str, List[str]],
        aggregation: str = "mean",
        require_all_ts_ids: bool = False,
        add_short_long_term_col: bool = False,
        add_dataset_col: bool = False,
        remove_outliers: bool = False,
        iqr_multiplier: float = 1.5,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieves a DataFrame of aggregated metrics for the specified metric, modes, and models.

        Parameters
        ----------
        metric : str
            The name of the metric column to aggregate (e.g., "mase", "rmsse").
        modes : Union[str, List[str]]
            A single user-friendly mode name (str) or a list of user-friendly mode names to include.
        models : Union[str, List[str]]
            A single user-friendly model name (str) or a list of user-friendly model names to include.
        aggregation : str, default="mean"
            The aggregation method to apply: "mean" or "median".
        require_all_ts_ids : bool, default=False
            If True, only includes `ts_id`s that have data for all selected model-mode
            combinations (and short/long term if applicable).
        add_short_long_term_col : bool, default=False
            If True, adds a column classifying `forecast_horizon` into 'short_term' (24, 7)
            or 'long_term' (168, 90).
        add_dataset_col : bool, default=False
            If True, adds the 'dataset_name' based on the 'sample_id'.
        remove_outliers : bool, default=False
            If True, removes outliers from the `metric` values based on IQR.
        iqr_multiplier : float, default=1.5
            Multiplier for the IQR to define the bounds for outlier removal.

        Returns
        -------
        Optional[pd.DataFrame]
            A DataFrame with aggregated metric values, or None if data cannot be loaded
            or filtered.
        """
        base_cols: List[str] = ["model", "mode", "ts_id", "sample_id"]
        # Required columns for loading depend on `add_short_long_term_col` and `add_dataset_col`
        required_cols_for_load: List[str] = base_cols + [metric, "forecast_horizon"]

        df_metrics: Optional[pd.DataFrame] = self._load_csv_file(
            self.metrics_output_path, required_cols=required_cols_for_load
        )

        if df_metrics is None or df_metrics.empty:
            logger.error("Metrics data not loaded or is empty. Cannot retrieve aggregated DataFrame.")
            return None

        # Ensure model and mode names are consistent (pretty names from self.model_map, self.mode_map)
        df_metrics['model'] = df_metrics['model'].map(self.model_map).fillna(df_metrics['model'])
        df_metrics['mode'] = df_metrics['mode'].map(self.mode_map).fillna(df_metrics['mode'])

        # Convert single string inputs to lists for consistent filtering
        modes_list = [modes] if isinstance(modes, str) else modes
        models_list = [models] if isinstance(models, str) else models

        if metric not in df_metrics.columns:
            logger.error(
                f"Metric '{metric}' not found in the loaded metrics data. "
                f"Available columns: {df_metrics.columns.tolist()}."
            )
            return None

        df_filtered: pd.DataFrame = df_metrics[
            (df_metrics["model"].isin(models_list)) & (df_metrics["mode"].isin(modes_list))
        ].copy()
        logger.info(f"Filtered metrics DataFrame to {len(df_filtered)} rows for models {models_list} and modes {modes_list}.")

        # Apply specific ts_id filters for "only_past" and "only_future" modes if included
        if "Only Past" in modes_list:
            df_filtered = df_filtered[df_filtered["ts_id"].isin(self.ts_ids_with_past_covs)]
        if "Only Future" in modes_list:
            df_filtered = df_filtered[df_filtered["ts_id"].isin(self.ts_ids_with_future_covs)]

        # Handle NaN and Inf values in the metric column
        if df_filtered[metric].isna().any() or np.isinf(df_filtered[metric]).any():
            initial_count = len(df_filtered)
            df_filtered[metric] = df_filtered[metric].replace([np.inf, -np.inf], np.nan)
            df_filtered.dropna(subset=[metric], inplace=True)
            if len(df_filtered) < initial_count:
                logger.warning(
                    f"Removed {initial_count - len(df_filtered)} rows with NaN/Inf values for metric '{metric}'."
                )

        if df_filtered.empty:
            logger.warning(
                f"No data found for the selected models ({models_list}) and modes ({modes_list}) "
                "after filtering for valid metric values."
            )
            return None

        # Define columns for grouping based on requested additions
        group_columns: List[str] = ["model", "mode", "ts_id"]

        if add_short_long_term_col:
            # Helper function to categorize forecast horizon
            def get_short_long_term(fh: int) -> Optional[str]:
                if fh in [24, 7]:
                    return "short_term"
                elif fh in [168, 90]:
                    return "long_term"
                return None

            df_filtered["short_long_term"] = df_filtered["forecast_horizon"].apply(get_short_long_term)
            df_filtered = df_filtered[df_filtered["short_long_term"].notna()].copy() # Remove rows not categorized
            group_columns.append("short_long_term")

            if df_filtered.empty:
                logger.warning(f"No data after categorizing into short/long term. Returning None.")
                return None

        if add_dataset_col:
            # Create a mapping from sample_id to dataset_name from DataLoader samples
            sample_to_dataset_map: Dict[int, str] = {s.id: s.dataset_name for s in self.dl.samples}
            df_filtered["dataset_name"] = df_filtered["sample_id"].map(sample_to_dataset_map)
            df_filtered.dropna(subset=['dataset_name'], inplace=True) # Drop if dataset_name is missing
            group_columns.append("dataset_name")
            if df_filtered.empty:
                logger.warning(f"No data after adding dataset_name and dropping NaNs. Returning None.")
                return None

        # Perform aggregation for each `ts_id` for the given `group_columns`
        if aggregation == "median":
            df_aggregated_per_ts = df_filtered.groupby(group_columns, as_index=False)[metric].median()
        elif aggregation == "mean":
            df_aggregated_per_ts = df_filtered.groupby(group_columns, as_index=False)[metric].mean()
        else:
            logger.error(f"Invalid aggregation method '{aggregation}'. Use 'mean' or 'median'.")
            return None

        logger.info(
            f"Aggregated metrics DataFrame has {len(df_aggregated_per_ts)} rows after per-ts aggregation."
        )

        # Apply `require_all_ts_ids` filter (if not already handled by dropna)
        if require_all_ts_ids:
            # Determine expected count of model-mode (and term if applicable) combinations per ts_id
            expected_combinations_count = len(models_list) * len(modes_list)
            if add_short_long_term_col:
                expected_combinations_count *= 2  # for 'short_term' and 'long_term'

            ts_id_counts = df_aggregated_per_ts.groupby("ts_id").size()
            ts_ids_to_include = ts_id_counts[
                ts_id_counts == expected_combinations_count
            ].index.tolist()
            df_aggregated_per_ts = df_aggregated_per_ts[
                df_aggregated_per_ts["ts_id"].isin(ts_ids_to_include)
            ].copy()

            if df_aggregated_per_ts.empty:
                logger.warning(
                    "After filtering for ts_ids available across all selected models/modes/terms, "
                    "no data remains. Returning None."
                )
                return None
            logger.info(
                f"Filtered metrics DataFrame to {len(df_aggregated_per_ts)} rows "
                "after applying require_all_ts_ids filter."
            )

        # Log final count of unique ts_ids included
        included_ts_ids_count = df_aggregated_per_ts["ts_id"].nunique()
        total_ts_ids_in_loader = len(self.dl.ts_ids)
        logger.info(
            f"Included {included_ts_ids_count} unique ts_ids out of {total_ts_ids_in_loader} "
            "total ts_ids in the final aggregated metrics DataFrame."
        )

        # Outlier removal (after per-ts aggregation, but before final group-by if any)
        if remove_outliers:
            initial_row_count = len(df_aggregated_per_ts)
            Q1 = df_aggregated_per_ts[metric].quantile(0.25)
            Q3 = df_aggregated_per_ts[metric].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR

            df_aggregated_per_ts = df_aggregated_per_ts[
                (df_aggregated_per_ts[metric] >= lower_bound)
                & (df_aggregated_per_ts[metric] <= upper_bound)
            ].copy()

            removed_count = initial_row_count - len(df_aggregated_per_ts)
            if removed_count > 0:
                logger.info(
                    f"Removed {removed_count} outlier metric values for '{metric}' "
                    "after per-ts aggregation."
                )
            else:
                logger.info(f"No outlier metric values removed for '{metric}'.")

        return df_aggregated_per_ts

    def plot_barplot(
        self,
        metric: str,
        modes: Union[str, List[str]],
        models: Union[str, List[str]],
        aggregation: str = "mean",
        require_all_ts_ids: bool = False,
        save_plot: bool = False,
        save_path: Optional[Path] = PLOTS_DIR,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        fig_size: Optional[Tuple[int, int]] = None,
        add_legend: bool = True,
        errorbar_type: str = "ci",
        sort_ascending: bool = True,
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Generates a HORIZONTAL bar plot for a specified metric, grouped by either model or mode,
        with error bars and an optional legend based on model categories.

        Parameters
        ----------
        metric : str
            The name of the metric to plot (e.g., "mase", "rmsse").
        modes : Union[str, List[str]]
            A single user-friendly mode name (str) or a list of user-friendly mode names to include.
        models : Union[str, List[str]]
            A single user-friendly model name (str) or a list of user-friendly model names to include.
        aggregation : str, default="mean"
            The aggregation method to use ("mean" or "median") for the `ts_id` level.
            The plotting function will then aggregate this aggregated data further.
        require_all_ts_ids : bool, default=False
            If True, only includes `ts_id`s that have data for all selected model-mode combinations.
        save_plot : bool, default=False
            Whether to save the plot to a file.
        save_path : Optional[Path], default=PLOTS_DIR
            Directory to save the plot.
        filename : Optional[str], default=None
            Custom filename for the saved plot. If None, a default filename is generated.
        title : Optional[str], default=None
            Custom title for the plot. If None, a default title is generated.
        fig_size : Optional[Tuple[int, int]], default=None
            Custom figure size (width, height) in inches. If None, size is determined dynamically.
        add_legend : bool, default=True
            If True, adds a legend (e.g., for model categories if models are compared).
        errorbar_type : str, default='ci'
            Type of error bar to display ('ci' for confidence interval, 'sd' for standard deviation, None for no error bars).
        sort_ascending : bool, default=True
            If True, bars are sorted in ascending order of metric value (best performance first).

        Returns
        -------
        Optional[Tuple[plt.Figure, plt.Axes]]
            A tuple containing the matplotlib Figure and Axes objects, or None if plotting fails.
        """
        # Ensure modes and models are lists for consistent processing
        _models_list = [models] if isinstance(models, str) else models
        _modes_list = [modes] if isinstance(modes, str) else modes

        # Retrieve aggregated data from get_aggregated_metrics_dataframe.
        # This DataFrame will contain one row per (model, mode, ts_id) combination,
        # with the specified aggregation applied to `forecast_horizon` if needed internally.
        df_plot_raw: Optional[pd.DataFrame] = self.get_aggregated_metrics_dataframe(
            metric=metric,
            modes=_modes_list,
            models=_models_list,
            aggregation=aggregation,  # This aggregation is applied at the ts_id level
            require_all_ts_ids=require_all_ts_ids,
            add_short_long_term_col=False, # Not needed for this plot type directly
            add_dataset_col=False # Not directly needed for grouping here
        )

        if df_plot_raw is None or df_plot_raw.empty:
            logger.error("Failed to retrieve data for plotting. Cannot generate bar plot.")
            return None

        df_plot: pd.DataFrame = df_plot_raw.dropna(subset=[metric]).copy()

        if df_plot.empty:
            logger.warning(
                f"After dropping rows with NaN in '{metric}', the DataFrame is empty. Cannot generate bar plot."
            )
            return None

        y_axis_col: Optional[str] = None
        hue_col: Optional[str] = None
        plot_legend: bool = False
        palette_to_use: Optional[Dict[str, str]] = None

        if len(_models_list) > 1 and len(_modes_list) == 1:
            y_axis_col = "model"
            if add_legend and self.model_categories:
                df_plot["model_category"] = df_plot["model"].map(self.model_categories).fillna("Other")
                hue_col = "model_category"
                plot_legend = True
                if self.category_colors:
                    actual_categories = df_plot["model_category"].unique()
                    palette_to_use = {
                        cat: self.category_colors.get(cat, "#808080") for cat in actual_categories
                    }
        elif len(_modes_list) > 1 and len(_models_list) == 1:
            y_axis_col = "mode"

            if add_legend and len(_models_list) > 1 and self.model_categories:
                df_plot["model_category"] = df_plot["model"].map(self.model_categories).fillna("Other")
                hue_col = "model_category"
                plot_legend = True
                if self.category_colors:
                    actual_categories = df_plot["model_category"].unique()
                    palette_to_use = {
                        cat: self.category_colors.get(cat, "#808080") for cat in actual_categories
                    }
        else:
            logger.error(
                "Invalid combination of models and modes for bar plot. Provide a list for models "
                "or modes to define the primary comparison axis."
            )
            return None

        if df_plot.empty or y_axis_col is None or y_axis_col not in df_plot.columns or df_plot[y_axis_col].nunique() == 0:
            logger.warning(f"Not enough data or valid configuration to plot after determining y-axis '{y_axis_col}'. Cannot plot.")
            return None
        
        # Calculate the aggregation for sorting the y-axis categories
        if aggregation == "mean":
            agg_values = df_plot.groupby(y_axis_col, observed=True)[metric].mean()
        else: # "median"
            agg_values = df_plot.groupby(y_axis_col, observed=True)[metric].median()
        
        sorted_categories: List[str] = agg_values.sort_values(ascending=sort_ascending).index.tolist()

        if fig_size is None:
            # Dynamic figure size based on number of categories
            fig_size = (10, max(5, len(sorted_categories) * 0.5))
        fig, ax = plt.subplots(figsize=fig_size, dpi=200)

        sns.barplot(
            data=df_plot,
            y=y_axis_col,
            x=metric,
            hue=hue_col,
            ax=ax,
            errorbar=errorbar_type,
            capsize=0.1,
            palette=palette_to_use,
            orient="h",
            order=sorted_categories,
        )

        if title is None:
            base_title = f"{aggregation.capitalize()} {metric.upper()} by {y_axis_col.replace('_', ' ').capitalize()}"
            if len(_models_list) == 1 and y_axis_col != "model":
                base_title += f" for Model: {_models_list[0]}"
            if len(_modes_list) == 1 and y_axis_col != "mode":
                base_title += f" in Mode: {_modes_list[0]}"
            title = base_title

        ax.set_title(title)
        ax.set_xlabel(f"{aggregation.capitalize()} {metric.upper()}")
        ax.set_ylabel(y_axis_col.replace("_", " ").capitalize())

        if plot_legend and hue_col and ax.get_legend() is not None:
            ax.legend(title=hue_col.replace("_", " ").capitalize(), loc="upper right")
            plt.tight_layout()
        elif ax.get_legend() is not None:
            ax.get_legend().remove()
        else:
            plt.tight_layout()

        if filename is None:
            # Create a more robust filename by joining lists or using placeholder if too long
            filename_models = "_".join(_models_list) if len(_models_list) <= 3 else "selected_models"
            filename_modes = "_".join(_modes_list) if len(_modes_list) <= 3 else "selected_modes"
            filename = f"barplot_{metric}_{filename_models}_{filename_modes}.png"

        self._plot_save_and_show(fig, save_path, filename, show_plot=True, save_plot=save_plot)
        return fig, ax

    def plot_model_improvement_by_dataset(
        self,
        model: str,
        metric: str,
        baseline_mode: str = "No Exogenous",
        comparison_mode: str = "Selected",
        forecast_term: Optional[str] = None,  # 'short_term', 'long_term', or None
        top_n_datasets: Optional[int] = None,  # Option to plot only top/bottom N datasets
        sort_ascending: bool = True,  # For top_n_datasets: True for best (lowest metric), False for worst improvement
        remove_outliers: bool = True,
        iqr_multiplier: float = 1.5,
        save_plot: bool = False,
        save_path: Optional[Path] = PLOTS_DIR,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        fig_size: Optional[Tuple[int, int]] = None,
        show_plot: bool = True,
        aggregation: str = "mean",  # How to aggregate ts_ids within a sample
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Plots the percentage improvement for a single model across different datasets (sample_ids).
        The improvement is calculated for a 'comparison_mode' relative to a 'baseline_mode'.

        Percentage Improvement = ((Baseline Metric - Comparison Metric) / Baseline Metric) * 100
        A positive value means the comparison mode performed better (lower metric value).

        Parameters
        ----------
        model : str
            The single user-friendly model name to analyze.
        metric : str
            The metric to evaluate (e.g., "mase", "rmsse").
        baseline_mode : str, default="No Exogenous"
            The user-friendly mode name to use as the baseline for comparison.
        comparison_mode : str, default="Selected"
            The single user-friendly mode name to compare against the baseline.
        forecast_term : Optional[str], default=None
            Filters the analysis to 'short_term' or 'long_term' forecasts. If None, considers all.
        top_n_datasets : Optional[int], default=None
            If provided, plots only the top N best or worst performing datasets.
        sort_ascending : bool, default=True
            For `top_n_datasets`: If True, sorts for best (lowest improvement); if False, for worst (highest improvement).
        remove_outliers : bool, default=True
            If True, removes datasets identified as outliers based on their improvement percentage.
        iqr_multiplier : float, default=1.5
            Multiplier for the IQR to define outlier bounds.
        save_plot : bool, default=False
            Whether to save the plot.
        save_path : Optional[Path], default=PLOTS_DIR
            Directory to save the plot.
        filename : Optional[str], default=None
            Custom filename for the saved plot. If None, a default filename is generated.
        title : Optional[str], default=None
            Custom title for the plot. If None, a default title is generated.
        fig_size : Optional[Tuple[int, int]], default=None
            Custom figure size (width, height) in inches. If None, size is determined dynamically.
        show_plot : bool, default=True
            Whether to display the plot.
        aggregation : str, default="mean"
            Method to aggregate metrics for `ts_id`s within a sample_id ("mean" or "median").

        Returns
        -------
        Optional[Tuple[plt.Figure, plt.Axes]]
            A tuple containing the matplotlib Figure and Axes objects, or None if plotting fails.
        """
        _models_list: List[str] = [model]
        all_relevant_modes: List[str] = [baseline_mode, comparison_mode]

        if forecast_term is not None and forecast_term not in ["short_term", "long_term"]:
            logger.error(
                f"Invalid 'forecast_term' '{forecast_term}'. Must be 'short_term', 'long_term', or None."
            )
            return None

        if aggregation not in ["mean", "median"]:
            logger.error(f"Invalid 'aggregation' '{aggregation}'. Must be 'mean' or 'median'.")
            return None

        # Determine if 'short_long_term' column is needed for `get_aggregated_metrics_dataframe`
        add_sl_col: bool = forecast_term is not None

        # Get aggregated metrics data, including dataset_name
        df_metrics_raw: Optional[pd.DataFrame] = self.get_aggregated_metrics_dataframe(
            metric=metric,
            modes=all_relevant_modes,
            models=_models_list,
            aggregation=aggregation,
            require_all_ts_ids=False,  # Don't apply this yet; consistency handled at sample level later
            add_short_long_term_col=add_sl_col,
            add_dataset_col=True,  # Crucially ensure dataset_name is added
        )

        if df_metrics_raw is None or df_metrics_raw.empty:
            logger.error("Metrics data is empty or None after initial load. Cannot plot model improvement by dataset.")
            return None

        # Filter by forecast_term if specified
        if forecast_term is not None:
            if "short_long_term" not in df_metrics_raw.columns:
                logger.error("'short_long_term' column not found, but 'forecast_term' was specified.")
                return None
            df_metrics_raw = df_metrics_raw[df_metrics_raw["short_long_term"] == forecast_term].copy()
            if df_metrics_raw.empty:
                logger.warning(f"No data found for forecast_term '{forecast_term}'. Cannot proceed.")
                return None

        # Ensure metric is numeric and clean up NaNs/Infs
        df_metrics_raw[metric] = pd.to_numeric(df_metrics_raw[metric], errors="coerce")
        df_metrics_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop rows where metric or dataset_name is NaN.
        df_metrics_raw.dropna(subset=[metric, "dataset_name"], inplace=True)

        if df_metrics_raw.empty:
            logger.warning("No valid metric or dataset_name data after initial filtering and cleaning. Cannot plot.")
            return None

        try:
            # Pivot to have dataset_name as index and modes as columns, with metric values
            pivot_df_datasets: pd.DataFrame = df_metrics_raw.pivot_table(
                index="dataset_name",
                columns="mode",
                values=metric,
            )
        except Exception as e:
            logger.error(f"Error pivoting aggregated dataset data for model '{model}' improvement: {e}")
            return None

        if pivot_df_datasets.empty:
            logger.warning(f"Pivoted data for model '{model}' is empty. Cannot calculate improvement by dataset.")
            return None

        if baseline_mode not in pivot_df_datasets.columns or comparison_mode not in pivot_df_datasets.columns:
            logger.error(
                f"Required modes ('{baseline_mode}' or '{comparison_mode}') not found in pivoted dataset data. "
                f"Available modes: {pivot_df_datasets.columns.tolist()}. Cannot calculate improvement."
            )
            return None

        # Calculate Percentage Improvement per dataset
        baseline_values: pd.Series = pivot_df_datasets[baseline_mode]
        comparison_values: pd.Series = pivot_df_datasets[comparison_mode]

        # Handle cases where baseline_values might be zero or very close to zero to avoid Inf results
        # Replace Inf with NaN, then drop NaNs
        improvement_pct: pd.Series = ((baseline_values - comparison_values) / baseline_values) * 100
        improvement_pct.replace([np.inf, -np.inf], np.nan, inplace=True)

        df_improvement_dataset: pd.DataFrame = pd.DataFrame(index=pivot_df_datasets.index)
        df_improvement_dataset["Percentage Improvement"] = improvement_pct
        df_improvement_dataset.dropna(subset=["Percentage Improvement"], inplace=True)

        if df_improvement_dataset.empty:
            logger.warning(
                f"No valid percentage improvement data for model '{model}' for {metric} in "
                f"{baseline_mode} vs {comparison_mode} after calculation and NaN removal."
            )
            return None

        # Outlier removal (on the improvement percentages across datasets)
        if remove_outliers:
            initial_dataset_count: int = len(df_improvement_dataset)
            Q1: float = df_improvement_dataset["Percentage Improvement"].quantile(0.25)
            Q3: float = df_improvement_dataset["Percentage Improvement"].quantile(0.75)
            IQR: float = Q3 - Q1
            lower_bound: float = Q1 - iqr_multiplier * IQR
            upper_bound: float = Q3 + iqr_multiplier * IQR

            df_improvement_dataset = df_improvement_dataset[
                (df_improvement_dataset["Percentage Improvement"] >= lower_bound)
                & (df_improvement_dataset["Percentage Improvement"] <= upper_bound)
            ].copy()

            removed_count: int = initial_dataset_count - len(df_improvement_dataset)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} outlier datasets for model '{model}' in improvement analysis.")
            else:
                logger.info(f"No outlier datasets removed for model '{model}'.")

        if df_improvement_dataset.empty:
            logger.warning(f"No data remaining after outlier removal for model '{model}'. Cannot plot.")
            return None

        df_plot_final: pd.DataFrame = df_improvement_dataset.reset_index()

        # Select top N datasets if specified
        if top_n_datasets is not None and top_n_datasets > 0:
            df_plot_final = df_plot_final.sort_values(
                by="Percentage Improvement", ascending=sort_ascending
            ).head(top_n_datasets).copy()

            if df_plot_final.empty:
                logger.warning(f"No data remaining after selecting top {top_n_datasets} datasets for model '{model}'.")
                return None

        # Sort for plotting (lowest improvement at top of bar chart if `sort_ascending` is True, i.e., good improvement)
        df_plot_sorted_for_viz: pd.DataFrame = df_plot_final.sort_values(by="Percentage Improvement", ascending=True)

        if fig_size is None:
            # Dynamic figure size based on number of datasets
            fig_size = (12, max(6, len(df_plot_sorted_for_viz) * 0.4))
        fig, ax = plt.subplots(figsize=fig_size, dpi=200)

        sns.barplot(
            data=df_plot_sorted_for_viz,
            x="Percentage Improvement",
            y="dataset_name",
            ax=ax,
            palette="coolwarm_r",  # _r reverses the colormap, so cool (blue) is positive (good)
            orient="h",
        )

        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No Improvement")

        if title is None:
            term_suffix: str = f" ({forecast_term.replace('_', ' ').capitalize()})" if forecast_term else ""
            title = (
                f"Percentage Improvement for {model} - {metric.upper()}{term_suffix}\n"
                f"({baseline_mode.capitalize()} Mode vs. {comparison_mode.capitalize()} Mode) per Dataset"
            )
        ax.set_title(title)
        ax.set_xlabel("Percentage Improvement (%)")
        ax.set_ylabel("Dataset Name")
        ax.legend(loc="lower right") # Add legend for "No Improvement" line

        plt.tight_layout()

        if filename is None:
            # Construct a descriptive filename
            term_str: str = f"_{forecast_term}" if forecast_term else ""
            filename = (
                f"barplot_model_improvement_by_dataset_{model}_{metric}_"
                f"{baseline_mode.replace(' ', '_')}_vs_{comparison_mode.replace(' ', '_')}{term_str}.png"
            )

        self._plot_save_and_show(fig, save_path, filename, show_plot, save_plot)
        return fig, ax

    def plot_short_long_term_comparison(
        self,
        metric: str,
        modes: Union[str, List[str]],
        models: Union[str, List[str]],
        aggregation: str = "mean",
        require_all_ts_ids: bool = False,
        save_plot: bool = False,
        save_path: Optional[Path] = PLOTS_DIR,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        fig_size: Optional[Tuple[int, int]] = None,
        add_legend: bool = True,
        errorbar_type: str = "ci",
        sort_by_metric: bool = True,  # Sort models/modes by overall metric value
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Generates a bar plot comparing short-term and long-term metric performance
        for specified models and modes. Each model/mode will have two bars: one for short-term
        and one for long-term forecasts.

        Parameters
        ----------
        metric : str
            The metric to plot (e.g., "mase", "rmsse").
        modes : Union[str, List[str]]
            A single user-friendly mode name (str) or a list of user-friendly mode names to include.
            Only modes that allow 'forecast_horizon' to be present and differentiated
            into short/long term should be used here.
        models : Union[str, List[str]]
            A single user-friendly model name (str) or a list of user-friendly model names to include.
        aggregation : str, default="mean"
            The aggregation method to use ("mean" or "median"). This aggregates `ts_id`s.
        require_all_ts_ids : bool, default=False
            If True, only includes `ts_id`s that have data for all
            selected models/modes AND short/long term categories.
        save_plot : bool, default=False
            Whether to save the plot to a file.
        save_path : Optional[Path], default=PLOTS_DIR
            Directory to save the plot.
        filename : Optional[str], default=None
            Custom filename for the saved plot. If None, a default filename is generated.
        title : Optional[str], default=None
            Custom title for the plot.
        fig_size : Optional[Tuple[int, int]], default=None
            Custom figure size (width, height) in inches. If None, size is determined dynamically.
        add_legend : bool, default=True
            If True, adds a legend for model categories (if comparing models).
        errorbar_type : str, default='ci'
            Type of error bar to display ('ci' for confidence interval, 'sd' for standard deviation, None for no error bars).
        sort_by_metric : bool, default=True
            If True, sorts the primary axis (model or mode) by the overall
            (short-term + long-term combined) aggregated metric value.

        Returns
        -------
        Optional[Tuple[plt.Figure, plt.Axes]]
            A tuple containing the matplotlib Figure and Axes objects, or None if plotting fails.
        """
        _modes_list = [modes] if isinstance(modes, str) else modes
        _models_list = [models] if isinstance(models, str) else models

        # Get aggregated data including the 'short_long_term' column
        df_plot_raw: Optional[pd.DataFrame] = self.get_aggregated_metrics_dataframe(
            metric=metric,
            modes=_modes_list,
            models=_models_list,
            aggregation=aggregation,
            require_all_ts_ids=require_all_ts_ids,
            add_short_long_term_col=True,  # Crucially request this column
            add_dataset_col=False # Not needed for this plot
        )

        if df_plot_raw is None or df_plot_raw.empty:
            logger.error("Failed to retrieve data for short/long term comparison. Cannot generate plot.")
            return None

        # Ensure 'short_long_term' column is present and valid
        if "short_long_term" not in df_plot_raw.columns:
            logger.error(
                "The 'short_long_term' column is missing from the aggregated DataFrame. "
                "This indicates an issue with 'get_aggregated_metrics_dataframe' or input data."
            )
            return None

        # Filter for actual 'short_term' and 'long_term' values if any None slipped through
        df_plot: pd.DataFrame = df_plot_raw[df_plot_raw["short_long_term"].notna()].copy()

        if df_plot.empty:
            logger.warning(
                f"No short or long term data found for metric '{metric}' with selected models and modes."
            )
            return None

        primary_axis_col: Optional[str] = None
        hue_col: str = "short_long_term"  # Always hue by short_long_term for this plot
        plot_legend_bool: bool = add_legend # Control legend visibility based on input parameter

        title_suffix: str = ""

        # Determine if we are comparing models or modes on the primary axis
        if len(_models_list) > 1 and len(_modes_list) >= 1:  # Compare models for selected mode(s)
            primary_axis_col = "model"
            title_suffix = f" (Modes: {', '.join(_modes_list)})"
            # Optional: Add model category as a secondary hue or for ordering
            if add_legend and self.model_categories:
                df_plot["model_category"] = df_plot["model"].map(self.model_categories).fillna("Other")
                # If a secondary hue, this would be a different sns.barplot call or complex.
                # Sticking to `short_long_term` as primary hue.
        elif len(_modes_list) > 1 and len(_models_list) == 1:  # Compare modes for a single model
            primary_axis_col = "mode"
            title_suffix = f" (Model: {_models_list[0]})"
        else:
            logger.error(
                "Invalid combination of models and modes for short/long term comparison. "
                "You must provide a list for either models (comparing models within one or more modes) "
                "or a list for modes (comparing modes for a single model)."
            )
            return None

        if primary_axis_col is None:
            return None

        # Sort the primary axis (models or modes) if requested
        sorted_primary_categories: List[str] = []
        if sort_by_metric:
            # Aggregate by primary_axis_col to get an overall order
            if aggregation == "mean":
                overall_agg = df_plot.groupby(primary_axis_col, observed=True)[metric].mean()
            else:  # median
                overall_agg = df_plot.groupby(primary_axis_col, observed=True)[metric].median()

            sorted_primary_categories = overall_agg.sort_values(ascending=True).index.tolist() 
        else:
            sorted_primary_categories = df_plot[primary_axis_col].unique().tolist()
            sorted_primary_categories.sort() # Ensure consistent alphabetical order if not by metric

        if fig_size is None:
            fig_size = (8, max(4, len(sorted_primary_categories) * 0.7)) # Dynamic height
        fig, ax = plt.subplots(figsize=fig_size, dpi=200)

        sns.barplot(
            data=df_plot,
            x=metric,
            y=primary_axis_col,
            hue=hue_col,
            ax=ax,
            errorbar=errorbar_type,
            capsize=0.1,
            palette="viridis",  # A good distinct palette for two categories (short/long term)
            orient="h",
            order=sorted_primary_categories,
        )

        # Set title and labels
        if title is None:
            base_title = (
                f"{aggregation.capitalize()} {metric.upper()} by "
                f"{primary_axis_col.replace('_', ' ').capitalize()} for Short-Term vs. Long-Term"
            )
            title = base_title + title_suffix

        ax.set_title(title)
        ax.set_xlabel(f"{aggregation.capitalize()} {metric.upper()}")
        ax.set_ylabel(primary_axis_col.replace("_", " ").capitalize())

        # Adjust legend if needed
        if plot_legend_bool and ax.get_legend() is not None:
            ax.legend(title="Forecast Horizon", loc="upper right", bbox_to_anchor=(1.02, 1)) # Outside axis
            plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout to make space for legend
        elif ax.get_legend() is not None:
            ax.get_legend().remove()
        else:
            plt.tight_layout() # No legend, just tight layout

        # Filename construction
        if filename is None:
            filename_models_str = "_".join(_models_list) if len(_models_list) <= 3 else "selected_models"
            filename_modes_str = "_".join(_modes_list) if len(_modes_list) <= 3 else "selected_modes"
            filename = f"barplot_short_long_term_{metric}_{filename_models_str}_{filename_modes_str}.png"

        self._plot_save_and_show(fig, save_path, filename, show_plot=True, save_plot=save_plot)
        return fig, ax

    def plot_time_series(
        self,
        ts_id: int,
        sample_id: int,
        models_to_plot: Optional[Union[str, List[str]]] = None,
        modes_to_plot: Optional[Union[str, List[str]]] = None,
        max_train_display_multiplier: int = 5,
        show_quantiles: bool = False,
        title: Optional[str] = None,
        fig_size: Optional[Tuple[int, int]] = None,
        filename_suffix: str = "",
        save_plot: bool = False,
        filename: Optional[str] = None,
        show_plot: bool = True,
        plot_save_dir: Path = PLOTS_DIR,
    ) -> None:
        """
        Plots a specific time series, including its historical data, actual future values,
        and optionally, forecasts from one or more models/modes.

        Parameters
        ----------
        ts_id : int
            The time series ID to plot.
        sample_id : int
            The sample ID associated with the time series and forecasts.
        models_to_plot : Optional[Union[str, List[str]]], default=None
            A single user-friendly model name or a list of model names to plot forecasts for.
        modes_to_plot : Optional[Union[str, List[str]]], default=None
            A single user-friendly mode name or a list of mode names to plot forecasts for.
            If `models_to_plot` is a list, `modes_to_plot` should be a single string.
            If `modes_to_plot` is a list, `models_to_plot` should be a single string.
        max_train_display_multiplier : int, default=5
            Multiplies the forecast horizon to determine how much historical training data to display.
        show_quantiles : bool, default=False
            If True, plots the lower and upper prediction interval bounds if available.
        title : Optional[str], default=None
            Custom title for the plot. If None, a default title is generated.
        fig_size : Optional[Tuple[int, int]], default=None
            Custom figure size (width, height) in inches. If None, (15, 7) is used.
        filename_suffix : str, default=""
            An additional suffix to append to the generated filename.
        save_plot : bool, default=False
            If True, saves the plot to a file.
        filename : Optional[str], default=None
            Custom filename for the saved plot. If None, a default filename is generated.
        show_plot : bool, default=True
            If True, displays the plot.
        plot_save_dir : Path, default=PLOTS_DIR
            The directory to save the plot.
        """
        if self.dl is None:
            logger.error("DataLoader (self.dl) not initialized. Cannot plot.")
            return

        plot_save_dir.mkdir(parents=True, exist_ok=True)
        if title is None:
            plot_title: str = f"Time Series {ts_id} (Sample ID: {sample_id})"
        else:
            plot_title = title
        filename_base: str = f"ts_plot_id_{ts_id}_sid{sample_id}"

        training_data_dict, actuals_dict = self.dl.get_forecast_actuals(sample_id)

        if ts_id not in actuals_dict:
            logger.error(f"ts_id {ts_id} not found in actuals_dict for sample_id {sample_id}.")
            return

        y_true_plot: np.ndarray = actuals_dict[ts_id]
        forecast_horizon_length: int = len(y_true_plot)

        y_train_plot: Optional[np.ndarray] = None
        if ts_id in training_data_dict and training_data_dict[ts_id] is not None:
            y_train_full: np.ndarray = training_data_dict[ts_id]
            history_len_to_plot: int = max_train_display_multiplier * forecast_horizon_length
            # Display only the most recent part of training data
            y_train_plot = y_train_full[-history_len_to_plot:]
        else:
            y_train_plot = np.array([]) # Empty array if no training data

        len_train_display: int = len(y_train_plot)
        x_train_indices: np.ndarray = np.arange(len_train_display)
        x_future_indices: np.ndarray = np.arange(len_train_display, len_train_display + forecast_horizon_length)

        if fig_size is None:
            fig_size = (15, 7)
        fig, ax = plt.subplots(figsize=fig_size, dpi=200)

        if len(y_train_plot) > 0:
            ax.plot(x_train_indices, y_train_plot, label="Historical Data", color="blue", linewidth=1.5)
        else:
            logger.warning(f"No training data available for ts_id {ts_id} in sample {sample_id}. Only plotting actuals and forecasts.")


        ax.plot(x_future_indices, y_true_plot, label="Actual Values", color="green", marker="o", markersize=4, linestyle="--", linewidth=1.5)
        
        forecast_specs_to_plot: List[Tuple[str, str]] = []
        if models_to_plot is not None and modes_to_plot is not None:
            m_list = [models_to_plot] if isinstance(models_to_plot, str) else models_to_plot
            d_list = [modes_to_plot] if isinstance(modes_to_plot, str) else modes_to_plot

            if len(m_list) == 1 and len(d_list) >= 1:
                # Plot multiple modes for a single model
                for mode_item in d_list:
                    forecast_specs_to_plot.append((m_list[0], mode_item))
            elif len(m_list) > 1 and len(d_list) == 1:
                # Plot multiple models for a single mode
                for model_item in m_list:
                    forecast_specs_to_plot.append((model_item, d_list[0]))
            elif len(m_list) == 1 and len(d_list) == 1:
                # Plot a single model-mode combination
                forecast_specs_to_plot.append((m_list[0], d_list[0]))
            else:
                logger.warning(
                    "Invalid combination for forecast plotting. Please provide either: "
                    "1. A single model and a list of modes. "
                    "2. A list of models and a single mode. "
                    "3. A single model and a single mode."
                )

        if forecast_specs_to_plot:
            if self.df_meta is None or self.df_data is None:
                logger.warning("Forecast metadata or data not loaded. Cannot plot forecasts.")
            else:
                # Generate a color palette for forecasts dynamically
                colors: List[Tuple[float, float, float, float]] = sns.color_palette("husl", len(forecast_specs_to_plot))
                for i, (model_name, mode_name) in enumerate(forecast_specs_to_plot):
                    # Find the metadata row for the specific model, mode, and sample_id
                    meta_row_filtered: pd.DataFrame = self.df_meta[
                        (self.df_meta["model"] == model_name)
                        & (self.df_meta["mode"] == mode_name)
                        & (self.df_meta["sample_id"] == sample_id)
                        & (self.df_meta["forecast_successful"])
                    ]
                    if meta_row_filtered.empty:
                        logger.warning(
                            f"No successful metadata found for model '{model_name}' in mode '{mode_name}' "
                            f"for sample_id {sample_id}. Skipping forecast plot for this combination."
                        )
                        continue

                    # Get the list of forecast IDs associated with this meta entry
                    forecast_ids_list: List[int] = meta_row_filtered.iloc[0]["forecast_ids_list"]
                    if not forecast_ids_list:
                        logger.warning(
                            f"No forecast IDs found in metadata for model '{model_name}' in mode '{mode_name}' "
                            f"for sample_id {sample_id}. Skipping forecast plot for this combination."
                        )
                        continue

                    # Retrieve the actual forecast arrays for the relevant ts_id
                    forecast_data_all_ts: Dict[int, Dict[str, np.ndarray]] = self._get_forecast_arrays(forecast_ids_list)
                    if ts_id not in forecast_data_all_ts:
                        logger.warning(
                            f"ts_id {ts_id} not found in forecast data for model '{model_name}' in mode '{mode_name}'. "
                            "Skipping forecast plot for this time series."
                        )
                        continue

                    fc_data: Dict[str, np.ndarray] = forecast_data_all_ts[ts_id]
                    y_pred: np.ndarray = fc_data["point"]
                    y_lower: Optional[np.ndarray] = fc_data.get("lower")
                    y_upper: Optional[np.ndarray] = fc_data.get("upper")

                    if len(y_pred) != forecast_horizon_length:
                        logger.warning(
                            f"Forecast length mismatch for {model_name}/{mode_name} on ts_id {ts_id}. "
                            f"Expected {forecast_horizon_length}, got {len(y_pred)}. Skipping forecast plot."
                        )
                        continue

                    # Plot point forecast
                    ax.plot(x_future_indices, y_pred, label=f"Forecast: {model_name} ({mode_name})", color=colors[i], linestyle="-", linewidth=1.5)
                    
                    # Plot prediction intervals if requested and available
                    if show_quantiles and y_lower is not None and y_upper is not None and len(y_lower) == forecast_horizon_length:
                        ax.fill_between(x_future_indices, y_lower, y_upper, color=colors[i], alpha=0.2, label=f"PI: {model_name} ({mode_name})")
                    elif show_quantiles and (y_lower is None or y_upper is None):
                        logger.info(f"Quantile data not available for {model_name} ({mode_name}) for ts_id {ts_id}. Skipping plotting prediction interval.")


        ax.set_title(plot_title)
        ax.set_xlabel("Time Step (Index)")
        ax.set_ylabel("Value")
        ax.legend(loc="best", fontsize="small")
        ax.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()

        # Construct filename
        if filename is None:
            # Dynamically add models and modes to filename
            model_names_str: str = "_".join(models_to_plot) if isinstance(models_to_plot, list) else (models_to_plot if models_to_plot else "no_models")
            mode_names_str: str = "_".join(modes_to_plot) if isinstance(modes_to_plot, list) else (modes_to_plot if modes_to_plot else "no_modes")
            filename = f"{filename_base}_{model_names_str}_{mode_names_str}{filename_suffix}.png"
            filename = filename.replace(" ", "_").replace("+", "plus").lower() # Sanitize filename for common characters


        self._plot_save_and_show(fig, plot_save_dir, filename, show_plot, save_plot)

    def plot_average_ranks_from_pivot(
        self,
        pivot_df: pd.DataFrame,
        metric: str,
        col_to_compare: str,
        fixed_context: str,
        add_legend: bool = True,
        save_plot: bool = False,
        save_path: Optional[Path] = PLOTS_DIR,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        fig_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Plots the average Friedman rank for each item (model or mode) based on a pivoted DataFrame.

        Parameters
        ----------
        pivot_df : pd.DataFrame
            A DataFrame where index is `ts_id` (or similar grouping unit), columns are items
            being compared (models or modes), and values are the metric scores.
        metric : str
            The name of the metric that was used for ranking (e.g., "mase").
        col_to_compare : str
            The name of the column that represents the items being compared (e.g., "model", "mode").
        fixed_context : str
            A string describing the fixed context of the comparison (e.g., "in Mode: All", "for Model: TimeGPT").
        add_legend : bool, default=True
            If True, adds a legend (e.g., for model categories if models are compared).
        save_plot : bool, default=False
            Whether to save the plot to a file.
        save_path : Optional[Path], default=PLOTS_DIR
            Directory to save the plot.
        filename : Optional[str], default=None
            Custom filename for the saved plot. If None, a default filename is generated.
        title : Optional[str], default=None
            Custom title for the plot. If None, a default title is generated.
        fig_size : Optional[Tuple[int, int]], default=None
            Custom figure size (width, height) in inches. If None, size is determined dynamically.
        """
        if pivot_df.empty:
            logger.warning("Pivot DataFrame is empty. Cannot plot average ranks.")
            return

        # Calculate ranks for each row (time series)
        ranks: np.ndarray = np.array([rankdata(row, method="average") for row in pivot_df.values])
        # Calculate the average rank for each column (model/mode)
        avg_ranks: np.ndarray = np.mean(ranks, axis=0)

        # Create a DataFrame for plotting
        plot_data: pd.DataFrame = pd.DataFrame(
            {
                col_to_compare: pivot_df.columns,
                "Average Rank": avg_ranks,
            }
        ).sort_values(by="Average Rank", ascending=True)

        hue_col: Optional[str] = None
        palette_to_use: Optional[Dict[str, str]] = None

        if col_to_compare == "model" and add_legend and self.model_categories:
            plot_data["model_category"] = plot_data[col_to_compare].map(self.model_categories).fillna("Other")
            hue_col = "model_category"
            if self.category_colors:
                actual_categories = plot_data["model_category"].unique()
                palette_to_use = {
                    cat: self.category_colors.get(cat, "#808080") for cat in actual_categories
                }

        if fig_size is None:
            # Dynamic figure size based on number of items being compared
            fig_size = (8, max(4, len(plot_data) * 0.5))
        fig, ax = plt.subplots(figsize=fig_size, dpi=200)

        sns.barplot(
            data=plot_data,
            y=col_to_compare,
            x="Average Rank",
            hue=hue_col,
            ax=ax,
            orient="h",
            palette=palette_to_use,
        )

        if title is None:
            title = f"Average Friedman Ranks {fixed_context}"

        ax.set_title(title)
        ax.set_xlabel("Average Friedman Rank")
        ax.set_ylabel(col_to_compare.replace("_", " ").capitalize())

        if hue_col and ax.get_legend() is not None:
            ax.legend(title=hue_col.replace("_", " ").capitalize(), loc="upper right")
            plt.tight_layout()
        elif ax.get_legend() is not None:
            ax.get_legend().remove()
        else:
            plt.tight_layout()

        if filename is None:
            # Sanitize fixed_context for filename
            sanitized_context = re.sub(r"[^\w\s-]", "", fixed_context).replace(" ", "_").lower()
            filename = f"friedman_ranks_{metric}_{sanitized_context}.png"
        
        self._plot_save_and_show(fig, save_path, filename, show_plot=True, save_plot=save_plot)

    def plot_friedman_analysis(
        self,
        metric: str,
        modes: Union[str, List[str]],
        models: Union[str, List[str]],
        aggregation: str = "mean",
        require_all_ts_ids: bool = False,
        forecast_term: Optional[str] = None,
        save_plot: bool = False,
        save_path: Optional[Path] = PLOTS_DIR,
        filename_rank: Optional[str] = None,
        filename_heatmap: Optional[str] = None,
        title_rank: Optional[str] = None,
        title_heatmap: Optional[str] = None,
        fig_size_rank: Optional[Tuple[int, int]] = None,
        fig_size_heatmap: Optional[Tuple[int, int]] = None,
        alpha: float = 0.05,
        post_hoc_method: str = "nemenyi",
        plot_ranks: bool = True,
        plot_heatmap: bool = True,
        add_legend: bool = True,
    ) -> None:
        """
        Performs a Friedman test to compare multiple models/modes over multiple time series
        and optionally plots average ranks and a p-value heatmap from post-hoc tests.

        Parameters
        ----------
        metric : str
            The metric to analyze (e.g., "mase", "rmsse").
        modes : Union[str, List[str]]
            A single user-friendly mode name (str) or a list of user-friendly mode names.
        models : Union[str, List[str]]
            A single user-friendly model name (str) or a list of user-friendly model names.
        aggregation : str, default="mean"
            The aggregation method to use ("mean" or "median") for `ts_id` level data.
        require_all_ts_ids : bool, default=False
            If True, only includes `ts_id`s that have data for all selected model-mode
            combinations (and forecast term if applicable).
        forecast_term : Optional[str], default=None
            Filters the analysis to 'short_term' or 'long_term' forecasts. If None, considers all.
        save_plot : bool, default=False
            Whether to save the generated plots.
        save_path : Optional[Path], default=PLOTS_DIR
            Directory to save the plots.
        filename_rank : Optional[str], default=None
            Custom filename for the average ranks plot.
        filename_heatmap : Optional[str], default=None
            Custom filename for the p-value heatmap.
        title_rank : Optional[str], default=None
            Custom title for the average ranks plot.
        title_heatmap : Optional[str], default=None
            Custom title for the p-value heatmap.
        fig_size_rank : Optional[Tuple[int, int]], default=None
            Figure size for the ranks plot.
        fig_size_heatmap : Optional[Tuple[int, int]], default=None
            Figure size for the heatmap.
        alpha : float, default=0.05
            Significance level for the Friedman test and post-hoc tests.
        post_hoc_method : str, default="nemenyi"
            The post-hoc test method to use if Friedman test is significant.
            Options: "nemenyi", "conover", "holm", "bonferroni" (for Conover-Iman with adjustment).
        plot_ranks : bool, default=True
            If True, generates and displays the average ranks bar plot.
        plot_heatmap : bool, default=True
            If True, generates and displays the post-hoc p-value heatmap (only if Friedman test is significant).
        add_legend : bool, default=True
            If True, adds a legend to the ranks plot for model categories.
        """
        _modes_list = [modes] if isinstance(modes, str) else modes
        _models_list = [models] if isinstance(models, str) else models

        if forecast_term is not None and forecast_term not in ["short_term", "long_term"]:
            logger.error(
                f"Invalid 'forecast_term' '{forecast_term}'. Must be 'short_term', 'long_term', or None."
            )
            return

        add_sl_col: bool = forecast_term is not None

        # Get aggregated metrics data
        df_metrics: Optional[pd.DataFrame] = self.get_aggregated_metrics_dataframe(
            metric=metric,
            modes=_modes_list,
            models=_models_list,
            aggregation=aggregation,
            require_all_ts_ids=require_all_ts_ids,
            add_short_long_term_col=add_sl_col,
            add_dataset_col=False # Not directly needed for this analysis's grouping
        )

        if df_metrics is None or df_metrics.empty:
            logger.error("Metrics data is empty or None. Cannot proceed with Friedman analysis.")
            return

        # Filter by forecast_term if specified
        if forecast_term is not None:
            df_metrics = df_metrics[df_metrics["short_long_term"] == forecast_term].copy()
            if df_metrics.empty:
                logger.warning(f"No data found for forecast_term '{forecast_term}'. Cannot proceed with Friedman analysis.")
                return
            df_metrics = df_metrics.drop(columns=["short_long_term"], errors="ignore")


        col_to_compare: str = ""
        fixed_context: str = ""
        if len(_models_list) > 1 and len(_modes_list) == 1:
            col_to_compare = "model"
            fixed_context = f"in Mode: {_modes_list[0]}"
            df_metrics = df_metrics.drop(columns=["mode"], errors="ignore")
        elif len(_modes_list) > 1 and len(_models_list) == 1:
            col_to_compare = "mode"
            fixed_context = f"for Model: {_models_list[0]}"
            df_metrics = df_metrics.drop(columns=["model"], errors="ignore")
        else:
            logger.error(
                "Invalid combination of modes and models for Friedman test. "
                "Provide a list for models (comparing models within one mode) "
                "or a list for modes (comparing modes for one model)."
            )
            return

        # Add forecast_term to fixed_context for clarity in plots and logs
        if forecast_term:
            fixed_context += f", Term: {forecast_term.replace('_term', '').capitalize()}"

        try:
            # Pivot the data to have `ts_id` as index and `col_to_compare` (models/modes) as columns
            pivot_df: pd.DataFrame = df_metrics.pivot_table(
                index="ts_id", columns=col_to_compare, values=metric
            )
        except Exception as e:
            logger.error(f"Error pivoting data for Friedman test: {e}")
            return

        original_rows: int = pivot_df.shape[0]
        # Drop rows (time series) with any missing values across the columns being compared
        pivot_df.dropna(inplace=True)
        dropped_rows: int = original_rows - pivot_df.shape[0]
        if dropped_rows > 0:
            logger.warning(
                f"Dropped {dropped_rows} time series due to missing data for one or more {col_to_compare}s "
                "in the pivoted DataFrame for Friedman test."
            )

        n_blocks: int = pivot_df.shape[0]  # Number of time series (blocks)
        n_treatments: int = pivot_df.shape[1]  # Number of models/modes (treatments)
        compared_items: List[str] = pivot_df.columns.tolist()

        if n_blocks < 2 or n_treatments < 2:
            logger.warning(
                f"Not enough data to perform Friedman test ({n_blocks} blocks, {n_treatments} treatments). "
                "Need at least 2 blocks and 2 treatments."
            )
            return

        # Plot average ranks before running Friedman test
        if plot_ranks:
            self.plot_average_ranks_from_pivot(
                pivot_df=pivot_df,
                metric=metric,
                col_to_compare=col_to_compare,
                fixed_context=fixed_context,
                add_legend=add_legend,
                save_plot=save_plot,
                save_path=save_path,
                title=title_rank,
                fig_size=fig_size_rank,
                filename=filename_rank,
            )

        # Perform Friedman test
        # `*data` unpacks a list of arrays (each column of pivot_df) into separate arguments
        # required by friedmanchisquare.
        data_for_friedman: List[np.ndarray] = [pivot_df[col].values for col in pivot_df.columns]
        stat, p_value = friedmanchisquare(*data_for_friedman)
        logger.info(
            f"Friedman Test for {metric} ({fixed_context}): Statistic={stat:.4f}, P-value={p_value:.4f}"
        )

        # Perform post-hoc test if Friedman test is significant
        if p_value < alpha:
            logger.info(
                f"Statistically significant difference found (p < {alpha}). Performing post-hoc test."
            )
            posthoc_results_df: Optional[pd.DataFrame] = None
            actual_posthoc_method: Optional[str] = None
            try:
                if post_hoc_method.lower() == "nemenyi":
                    posthoc_results_df = sp.posthoc_nemenyi_friedman(pivot_df)
                    actual_posthoc_method = "Nemenyi"
                elif post_hoc_method.lower() in ["conover", "holm", "bonferroni"]:
                    adj_method = post_hoc_method.lower()
                    posthoc_results_df = sp.posthoc_conover_friedman(pivot_df, p_adjust=adj_method)
                    actual_posthoc_method = f"Conover-Iman ({adj_method.capitalize()})"
                else:
                    logger.warning(
                        f"Unsupported post-hoc method '{post_hoc_method}'. Skipping post-hoc test."
                    )

            except Exception as e:
                logger.error(f"Error during post-hoc test ({post_hoc_method}): {e}")

            if posthoc_results_df is not None:
                significant_pairs_list: List[Tuple[str, str, float]] = []
                # Iterate through the lower triangle of the p-value matrix
                for i, item1 in enumerate(posthoc_results_df.columns):
                    for j, item2 in enumerate(posthoc_results_df.index):
                        # Ensure we only check each pair once (e.g., item1 vs item2, not item2 vs item1)
                        # and not self-comparisons
                        if i < j and posthoc_results_df.loc[item2, item1] < alpha:
                            significant_pairs_list.append((item1, item2, posthoc_results_df.loc[item2, item1]))

                if significant_pairs_list:
                    # Sort by p-value for clearer output
                    significant_pairs_list.sort(key=lambda x: x[2])
                    logger.info(
                        f"Significant differences (pairwise p < {alpha}) found between the following {col_to_compare} pairs:"
                    )
                    for m1, m2, p in significant_pairs_list:
                        logger.info(f"  - {m1} vs {m2} (p={p:.4f})")
                else:
                    logger.info(f"No significant differences found between specific {col_to_compare} pairs.")

                # Plot p-value heatmap
                if plot_heatmap:
                    n_items: int = len(compared_items)
                    if fig_size_heatmap is None:
                        fig_size_heatmap = (max(6, n_items * 0.8), max(4, n_items * 0.6))
                    fig, ax = plt.subplots(figsize=fig_size_heatmap, dpi=200)

                    sns.heatmap(
                        posthoc_results_df.reindex(index=compared_items, columns=compared_items),
                        annot=True,
                        fmt=".2f",
                        cmap="Blues_r",  # Reversed Blues cmap: darker blue for smaller p-values
                        ax=ax,
                        cbar_kws={"label": "Pairwise p-value"},
                        linewidths=0.5,
                        linecolor="lightgray",
                    )

                    ax.set_xticks(np.arange(n_items) + 0.5)
                    ax.set_xticklabels(compared_items, rotation=90, ha="right")
                    ax.set_yticks(np.arange(n_items) + 0.5)
                    ax.set_yticklabels(compared_items)
                    ax.tick_params(axis='x', labelbottom=True) # Ensure x-axis labels are visible
                    ax.tick_params(axis='y', labelleft=True) # Ensure y-axis labels are visible

                    if title_heatmap is None:
                        # Attempt to make a more concise title
                        cut_context: str = re.sub(r"^.*?(Mode)", r"\1", fixed_context)
                        title_heatmap = f"{actual_posthoc_method} Pairwise p-values for {col_to_compare.capitalize()}\n{cut_context}"
                    ax.set_title(title_heatmap)
                    plt.tight_layout()

                    if filename_heatmap is None:
                        # Sanitize filename
                        sanitized_context_hm = re.sub(r"[^\w\s-]", "", fixed_context).replace(" ", "_").lower()
                        filename_heatmap = f"friedman_heatmap_{metric}_{sanitized_context_hm}.png"
                    self._plot_save_and_show(fig, save_path, filename_heatmap, show_plot=True, save_plot=save_plot)
        else:
            logger.info(f"No statistically significant difference detected (p >= {alpha}). No post-hoc test performed.")

    def plot_metric_completeness(
        self,
        metric: str,
        modes: Union[str, List[str]],
        models: Union[str, List[str]],
        save_plot: bool = False,
        save_path: Optional[Path] = PLOTS_DIR,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        fig_size: Optional[Tuple[int, int]] = None,
        show_plot: bool = True
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Plots the percentage of ts_ids for which a given metric has valid data (non-NaN/Inf)
        for each model-mode combination, relative to the total number of ts_ids in the DataLoader.

        Parameters
        ----------
        metric : str
            The metric to check completeness for (e.g., "mase").
        modes : Union[str, List[str]]
            A single user-friendly mode name (str) or a list of mode names to include.
        models : Union[str, List[str]]
            A single user-friendly model name (str) or a list of model names to include.
        save_plot : bool, default=False
            If True, saves the generated plot image.
        save_path : Optional[Path], default=PLOTS_DIR
            Path to save the generated plot image.
        filename : Optional[str], default=None
            Custom filename for the saved plot. If None, a default filename is generated.
        title : Optional[str], default=None
            Custom title for the bar plot.
        fig_size : Optional[Tuple[int, int]], default=None
            Custom figure size (width, height) in inches. If None, size is determined dynamically.
        show_plot : bool, default=True
            Whether to display the plot.

        Returns
        -------
        Optional[Tuple[plt.Figure, plt.Axes]]
            A tuple containing the matplotlib Figure and Axes objects, or None if plotting fails.
        """
        # Load df_metrics if it's not already loaded, ensuring it has required columns
        df_metrics_for_completeness: Optional[pd.DataFrame] = self.df_metrics
        if df_metrics_for_completeness is None:
            # Explicitly load with required columns if self.df_metrics wasn't fully loaded
            df_metrics_for_completeness = self._load_csv_file(
                self.metrics_output_path, required_cols=["model", "mode", "ts_id", metric, "sample_id", "forecast_horizon"]
            )
            if df_metrics_for_completeness is not None:
                df_metrics_for_completeness['model'] = df_metrics_for_completeness['model'].map(self.model_map).fillna(df_metrics_for_completeness['model'])
                df_metrics_for_completeness['mode'] = df_metrics_for_completeness['mode'].map(self.mode_map).fillna(df_metrics_for_completeness['mode'])

        if df_metrics_for_completeness is None or df_metrics_for_completeness.empty:
            logger.error("Metrics data not loaded or is empty. Cannot plot metric completeness.")
            return None

        _modes_list = [modes] if isinstance(modes, str) else modes
        _models_list = [models] if isinstance(models, str) else models

        if metric not in df_metrics_for_completeness.columns:
            logger.error(f"Metric '{metric}' not found in the metrics data. Available columns: {df_metrics_for_completeness.columns.tolist()}.")
            return None
        
        df_filtered_for_completeness: pd.DataFrame = df_metrics_for_completeness[
            df_metrics_for_completeness['model'].isin(_models_list) &
            df_metrics_for_completeness['mode'].isin(_modes_list)
        ].copy()

        if df_filtered_for_completeness.empty:
            logger.warning("No data found for the selected models and modes to plot metric completeness.")
            return None

        df_completeness_data: List[Dict[str, Union[str, float]]] = []
        total_dl_ts_ids: int = len(self.dl.ts_ids)

        for (model, mode), group in df_filtered_for_completeness.groupby(['model', 'mode']):
            # Filter out NaN/Inf values for the specific metric. Use .nunique() to count unique ts_ids with valid data.
            valid_metric_ts_ids: int = group[metric].replace([np.inf, -np.inf], np.nan).dropna()['ts_id'].nunique()
            
            completeness_percentage: float = (valid_metric_ts_ids / total_dl_ts_ids) * 100 if total_dl_ts_ids > 0 else 0.0
            
            df_completeness_data.append({
                'model': model,
                'mode': mode,
                'Metric Completeness (%)': completeness_percentage,
                'Key': f"{model} ({mode})" # Combine for single bar label
            })
        
        plot_data: pd.DataFrame = pd.DataFrame(df_completeness_data).sort_values(by='Metric Completeness (%)', ascending=False)
        
        if fig_size is None:
            fig_size = (12, max(6, len(plot_data) * 0.5))
        fig, ax = plt.subplots(figsize=fig_size, dpi=200)

        sns.barplot(data=plot_data, y='Key', x='Metric Completeness (%)', ax=ax, palette='viridis', orient='h')

        ax.set_xlim(0, 100) # Ensure percentages are within 0-100
        ax.set_xlabel("Metric Completeness (%)")
        ax.set_ylabel("Model (Mode) Combination")

        if title is None:
            title = f"Metric Completeness (%) for '{metric}' across Models and Modes"
        ax.set_title(title)
        
        plt.tight_layout()

        if filename is None:
            filename_models_str = "_".join(_models_list) if len(_models_list) <= 3 else "selected_models"
            filename_modes_str = "_".join(_modes_list) if len(_modes_list) <= 3 else "selected_modes"
            filename = f"metric_completeness_{metric}_{filename_models_str}_{filename_modes_str}.png"
        self._plot_save_and_show(fig, save_path, filename, show_plot, save_plot)
        return fig, ax
    
    def plot_percentage_improvement_boxplots(
        self,
        metric: str,
        models: Union[str, List[str]],
        baseline_mode: str = "No Exogenous",
        comparison_mode: str = "Selected with Lags",
        forecast_term: Optional[str] = None,
        require_all_ts_ids: bool = False,
        aggregation: str = "mean",
        remove_outliers: bool = True,
        iqr_multiplier: float = 1.5,
        save_plot: bool = False,
        save_path: Optional[Path] = PLOTS_DIR,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        fig_size: Optional[Tuple[int, int]] = None,
        show_plot: bool = True,
        sort_by_median_improvement: bool = True
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Generates boxplots showing the percentage improvement of forecasts from specified
        models using a 'comparison_mode' relative to a 'baseline_mode'.

        Percentage Improvement = ((Baseline Metric - Comparison Metric) / Baseline Metric) * 100
        A positive value means the comparison mode performed better (lower metric).

        Parameters
        ----------
        metric : str
            The metric to evaluate (e.g., "mase", "rmsse").
        models : Union[str, List[str]]
            Single user-friendly model name (str) or list of model names to include.
        baseline_mode : str, default="No Exogenous"
            The mode to use as the baseline for comparison.
        comparison_mode : str, default="Selected with Lags"
            The single mode to compare against the baseline.
        forecast_term : Optional[str], default=None
            Filters the analysis to 'short_term' or 'long_term' forecasts.
        require_all_ts_ids : bool, default=False
            If True, only includes ts_ids that have data for all
            selected models, all relevant modes, and (if applicable)
            both short/long term categories for consistency.
        aggregation : str, default="mean"
            Method to aggregate `ts_id`s within a sample_id for metric calculation ("mean" or "median").
        remove_outliers : bool, default=True
            If True, removes outliers from the improvement percentages.
            If a ts_id's improvement is an outlier for *any* model in the plot, that ts_id is removed for *all* models.
        iqr_multiplier : float, default=1.5
            Multiplier for the IQR to define outlier bounds (default: 1.5 for boxplot whiskers).
        save_plot : bool, default=False
            Whether to save the plot to a file.
        save_path : Optional[Path], default=PLOTS_DIR
            Directory to save the plot.
        filename : Optional[str], default=None
            Custom filename for the saved plot.
        title : Optional[str], default=None
            Custom title for the plot.
        fig_size : Optional[Tuple[int, int]], default=None
            Custom figure size (width, height) in inches.
        show_plot : bool, default=True
            Whether to display the plot.
        sort_by_median_improvement : bool, default=True
            If True, sorts models by their median percentage improvement (descending for positive improvement).

        Returns
        -------
        Optional[Tuple[plt.Figure, plt.Axes]]
            A tuple containing the matplotlib Figure and Axes objects, or None if plotting fails.
        """
        _models_list = [models] if isinstance(models, str) else models
        
        all_relevant_modes: List[str] = list(set([baseline_mode, comparison_mode]))

        if not all_relevant_modes:
            logger.error("No modes specified for comparison.")
            return None

        if forecast_term is not None and forecast_term not in ['short_term', 'long_term']:
            logger.error(f"Invalid 'forecast_term' '{forecast_term}'. Must be 'short_term', 'long_term', or None.")
            return None

        add_sl_col: bool = (forecast_term is not None)

        df_metrics: Optional[pd.DataFrame] = self.get_aggregated_metrics_dataframe(
            metric=metric,
            modes=all_relevant_modes,
            models=_models_list,
            aggregation=aggregation,
            require_all_ts_ids=require_all_ts_ids,
            add_short_long_term_col=add_sl_col,
            add_dataset_col=False # Not directly needed for this plot
        )
        
        if df_metrics is None or df_metrics.empty:
            logger.error("Metrics data is empty or None. Cannot calculate percentage improvement.")
            return None

        if forecast_term is not None:
            if 'short_long_term' not in df_metrics.columns:
                logger.error("'short_long_term' column not found, but 'forecast_term' was specified.")
                return None
            df_metrics = df_metrics[df_metrics['short_long_term'] == forecast_term].copy()
            if df_metrics.empty:
                logger.warning(f"No data found for forecast_term '{forecast_term}'. Cannot proceed.")
                return None
        
        df_metrics[metric] = pd.to_numeric(df_metrics[metric], errors='coerce')
        df_metrics.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_metrics.dropna(subset=[metric], inplace=True)
        
        if df_metrics.empty:
            logger.warning("No valid metric data after initial filtering and NaN/Inf removal. Cannot plot.")
            return None

        # Pivot to get metrics for baseline and comparison modes side-by-side per model-ts_id
        try:
            # Create a unique identifier for each model-ts_id combination
            # This is important for ensuring correct calculation of improvement *per ts_id for each model*.
            df_metrics['model_ts_id'] = df_metrics['model'].astype(str) + '_' + df_metrics['ts_id'].astype(str)
            
            pivot_df: pd.DataFrame = df_metrics.pivot_table(index='model_ts_id', columns='mode', values=metric)
        except Exception as e:
            logger.error(f"Error pivoting data for improvement calculation: {e}")
            return None

        if pivot_df.empty:
            logger.warning("Pivoted DataFrame is empty. Cannot calculate improvement.")
            return None

        if baseline_mode not in pivot_df.columns:
            logger.error(f"Baseline mode '{baseline_mode}' not found in the data after pivoting. Check data or baseline_mode parameter.")
            return None
        if comparison_mode not in pivot_df.columns:
            logger.error(f"Comparison mode '{comparison_mode}' not found in the data after pivoting. Check data or comparison_mode parameter.")
            return None

        improvement_data_list: List[Dict[str, Union[str, int, float]]] = []
        for model_ts_id, row in pivot_df.iterrows():
            parts = model_ts_id.split('_')
            # Handle model names that might contain underscores (e.g., "Chronos + LR")
            model_name = '_'.join(parts[:-1]) 
            ts_id = int(parts[-1])
            
            baseline_value = row.get(baseline_mode)
            comparison_value = row.get(comparison_mode)

            # Skip if either value is NaN or baseline is zero (to avoid division by zero/inf)
            if pd.isna(baseline_value) or pd.isna(comparison_value) or baseline_value == 0:
                continue

            improvement_pct: float = ((baseline_value - comparison_value) / baseline_value) * 100
            
            improvement_data_list.append({
                'model': model_name,
                'ts_id': ts_id,
                'Percentage Improvement': improvement_pct
            })

        df_improvement: pd.DataFrame = pd.DataFrame(improvement_data_list)

        if df_improvement.empty:
            logger.warning("No percentage improvement data generated. This might be due to missing baseline or comparison data for common ts_ids.")
            return None

        # --- Outlier Removal Logic (modified to remove ts_ids that are outliers for ANY model) ---
        if remove_outliers:
            initial_ts_ids_count = df_improvement['ts_id'].nunique()
            outlier_ts_ids_set = set()

            # Identify all ts_ids that are outliers for *any* model in the current selection
            for model_name, group in df_improvement.groupby('model'):
                Q1 = group['Percentage Improvement'].quantile(0.25)
                Q3 = group['Percentage Improvement'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                # Find ts_ids that fall outside the bounds for this specific model's distribution
                model_outliers = group[(group['Percentage Improvement'] < lower_bound) | 
                                       (group['Percentage Improvement'] > upper_bound)]
                outlier_ts_ids_set.update(model_outliers['ts_id'].tolist())

            # Filter the entire DataFrame to remove all rows (for all models) associated with identified outlier ts_ids
            if outlier_ts_ids_set:
                df_improvement_original_row_count = len(df_improvement)
                df_improvement = df_improvement[~df_improvement['ts_id'].isin(outlier_ts_ids_set)].copy()
                removed_rows_count = df_improvement_original_row_count - len(df_improvement)
                
                logger.info(f"Removed {len(outlier_ts_ids_set)} unique ts_ids (total {removed_rows_count} rows) "
                            f"identified as outliers for at least one model across '{metric}'. "
                            f"Remaining unique ts_ids: {df_improvement['ts_id'].nunique()} out of {initial_ts_ids_count}.")
            else:
                logger.info("No ts_ids identified as outliers across any model. No data removed.")

        if df_improvement.empty:
            logger.warning("No data remaining after outlier removal. Cannot plot.")
            return None

        # Sort models for plotting (by median improvement)
        model_order: Optional[List[str]] = None
        if sort_by_median_improvement:
            median_improvements = df_improvement.groupby('model')['Percentage Improvement'].median().sort_values(ascending=False)
            model_order = median_improvements.index.tolist()
        else:
            model_order = sorted(df_improvement['model'].unique())
        
        if fig_size is None:
            fig_size = (14, max(8, len(model_order) * 0.7)) # Dynamic height based on number of models
        fig, ax = plt.subplots(figsize=fig_size, dpi=200)

        # Plot boxplots
        sns.boxplot(
            data=df_improvement,
            x='Percentage Improvement',
            y='model',
            order=model_order,
            ax=ax,
            palette='viridis',
            orient='h',
            showfliers=False # Outliers are explicitly removed in preprocessing, so don't show them here.
        )

        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No Improvement')

        # Set title and labels
        if title is None:
            term_suffix: str = f" ({forecast_term.replace('_', ' ').capitalize()})" if forecast_term else ""
            title = (
                f"Percentage Improvement for {metric.upper()}{term_suffix}\n"
                f"({baseline_mode.capitalize()} Mode vs. {comparison_mode.capitalize()} Mode)"
            )
        ax.set_title(title)
        ax.set_xlabel("Percentage Improvement (%)")
        ax.set_ylabel("Model")
        
        ax.legend(loc='lower right') # Legend for 'No Improvement' line

        plt.tight_layout()
        
        # Filename construction
        models_str = '_'.join(_models_list) if len(_models_list) <= 3 else "selected_models"
        term_str = f"_{forecast_term}" if forecast_term else ""
        if filename is None:
            filename = f"boxplot_improvement_{metric}_{baseline_mode.replace(' ', '_')}_vs_{comparison_mode.replace(' ', '_')}_{models_str}{term_str}.png"
            filename = filename.replace(" ", "_").replace("+", "plus").lower() # Sanitize filename

        self._plot_save_and_show(fig, save_path, filename, show_plot, save_plot)
        return fig, ax

    def get_top_n_ts_ids(
        self,
        metric: str,
        models: Union[str, List[str]],
        modes: Union[str, List[str]],
        n: int = 3,
        best: bool = True,
        plot_results: bool = True,
        plot_save_dir: Path = PLOTS_DIR,
        save_plot: bool = False,
    ) -> List[Tuple[int, int, float]]:
        """
        Retrieves the top N best or worst performing time series (ts_id, sample_id) pairs
        based on a specific metric for given models and modes. Ensures unique ts_ids.

        Parameters
        ----------
        metric : str
            The metric to evaluate (e.g., "mase", "rmsse").
        models : Union[str, List[str]]
            Single user-friendly model name (str) or list of model names.
        modes : Union[str, List[str]]
            Single user-friendly mode name (str) or list of mode names.
        n : int, default=3
            The number of unique `ts_id`s to retrieve.
        best : bool, default=True
            If True, returns the best performing `ts_id`s (lowest metric value).
            If False, returns the worst performing `ts_id`s (highest metric value).
        plot_results : bool, default=True
            If True, plots each of the retrieved time series using `plot_time_series`.
        plot_save_dir : Path, default=PLOTS_DIR
            Directory to save plots if `plot_results` is True.
        save_plot : bool, default=False
            Whether to save the individual time series plots.

        Returns
        -------
        List[Tuple[int, int, float]]
            A list of tuples (`ts_id`, `sample_id`, `metric_value`) for the top N unique `ts_id`s.
            Returns an empty list if no valid data is found.
        """
        # Load df_metrics if it's not already loaded, ensuring it has required columns
        df_metrics_for_top_n: Optional[pd.DataFrame] = self.df_metrics
        if df_metrics_for_top_n is None:
            df_metrics_for_top_n = self._load_csv_file(
                self.metrics_output_path, required_cols=["model", "mode", "ts_id", metric, "sample_id", "forecast_horizon"]
            )
            if df_metrics_for_top_n is not None:
                df_metrics_for_top_n['model'] = df_metrics_for_top_n['model'].map(self.model_map).fillna(df_metrics_for_top_n['model'])
                df_metrics_for_top_n['mode'] = df_metrics_for_top_n['mode'].map(self.mode_map).fillna(df_metrics_for_top_n['mode'])


        if df_metrics_for_top_n is None or df_metrics_for_top_n.empty:
            logger.error("Raw metrics data not loaded or is empty. Cannot retrieve top N ts_ids.")
            return []

        _models_list = [models] if isinstance(models, str) else models
        _modes_list = [modes] if isinstance(modes, str) else modes

        if metric not in df_metrics_for_top_n.columns:
            logger.error(f"Metric '{metric}' not found in the metrics data. Available columns: {df_metrics_for_top_n.columns.tolist()}.")
            return []

        df_filtered_for_top_n: pd.DataFrame = df_metrics_for_top_n[
            df_metrics_for_top_n['model'].isin(_models_list) &
            df_metrics_for_top_n['mode'].isin(_modes_list)
        ].copy()

        df_filtered_for_top_n[metric] = pd.to_numeric(df_filtered_for_top_n[metric], errors='coerce')
        df_filtered_for_top_n.dropna(subset=[metric], inplace=True)
        
        if df_filtered_for_top_n.empty:
            logger.warning(f"No valid data for metric '{metric}' found for the specified models and modes.")
            return []

        # Sort by metric value to get best/worst
        sort_ascending: bool = best # For 'best', we want lowest metric value (ascending). For 'worst', highest (descending).
        df_sorted: pd.DataFrame = df_filtered_for_top_n.sort_values(by=metric, ascending=sort_ascending).reset_index(drop=True)

        selected_ts_ids: List[int] = []
        results: List[Tuple[int, int, float]] = []

        # Iterate and collect unique ts_ids until N are found
        for _, row in df_sorted.iterrows():
            ts_id: int = int(row['ts_id'])
            sample_id: int = int(row['sample_id'])
            metric_value: float = float(row[metric])

            if ts_id not in selected_ts_ids:
                selected_ts_ids.append(ts_id)
                results.append((ts_id, sample_id, metric_value))
            
            if len(selected_ts_ids) >= n:
                break
        
        if not results:
            logger.warning(f"Could not find {n} unique ts_ids with valid data for metric '{metric}'. Found {len(selected_ts_ids)}.")
            return []

        logger.info(f"Retrieved {len(results)} unique ts_id/sample_id pairs for {('best' if best else 'worst')} '{metric}':")
        for ts_id, sample_id, value in results:
            logger.info(f"  TS_ID: {ts_id}, Sample_ID: {sample_id}, {metric}: {value:.4f}")

        if plot_results:
            logger.info(f"Plotting results for the {('best' if best else 'worst')} {len(results)} ts_ids.")
            for ts_id_to_plot, sample_id_to_plot, _ in results:
                self.plot_time_series(
                    ts_id=ts_id_to_plot,
                    sample_id=sample_id_to_plot,
                    models_to_plot=_models_list, # Plot all specified models
                    modes_to_plot=_modes_list, # Plot all specified modes
                    save_plot=save_plot,
                    show_plot=True,
                    plot_save_dir=plot_save_dir,
                    filename_suffix=f"_{metric}_{'best' if best else 'worst'}_overall"
                )
        
        return results

    def plot_metric_heatmap_by_dataset(
        self,
        metric: str,
        modes: Union[str, List[str]],
        models_to_plot: List[str],
        datasets_as_cols: Optional[List[str]] = None,
        metric_aggregation: str = "mean",
        lower_metric_is_better: bool = True,
        max_auto_derived_datasets: int = 20,
        keep_all_specified_models: bool = False,
        annotation_fmt: str = ".2f",
        save_plot: bool = False,
        save_path_base: Optional[Path] = PLOTS_DIR,
        filename: Optional[str] = None,
        show_plot: bool = True,
        title: Optional[str] = None,
        fig_size: Optional[Tuple[int, int]] = None
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Generates a heatmap of aggregated metrics, with models as rows and datasets as columns.
        Coloring is relative within each dataset column. Default colormap is blue (best) to white (worst).
        Colorbar ticks are customized for clarity.

        Parameters
        ----------
        metric : str
            The performance metric to use (e.g., "mase").
        modes : Union[str, List[str]]
            Mode(s) to filter metric data.
        models_to_plot : List[str]
            List of model names (heatmap rows).
        datasets_as_cols : Optional[List[str]], default=None
            Optional list of dataset names (heatmap columns). If None, inferred from data.
        metric_aggregation : str, default="mean"
            Aggregation ("mean" or "median") for metric values within each model-dataset-mode cell.
        lower_metric_is_better : bool, default=True
            True if a lower metric value indicates better performance.
        max_auto_derived_datasets : int, default=20
            Maximum number of dataset columns to auto-derive before a warning is logged.
        keep_all_specified_models : bool, default=False
            Strategy for handling missing data. If True, all `models_to_plot` are kept,
            and only dataset columns with complete data for these models are shown.
            If False, all dataset columns with any data are kept, and models with
            missing data in those columns are dropped.
        annotation_fmt : str, default=".2f"
            Format string for cell annotations (original metric values).
        save_plot : bool, default=False
            Whether to save the plot.
        save_path_base : Optional[Path], default=PLOTS_DIR
            Base directory for saving plots.
        filename : Optional[str], default=None
            Custom filename for the saved plot. If None, a default filename is generated.
        show_plot : bool, default=True
            Whether to display the plot.
        title : Optional[str], default=None
            Optional custom title for the plot.
        fig_size : Optional[Tuple[int, int]], default=None
            Custom figure size (width, height) in inches.

        Returns
        -------
        Optional[Tuple[plt.Figure, plt.Axes]]
            Matplotlib Figure and Axes objects, or None if plotting fails.
        """
        _modes_list = [modes] if isinstance(modes, str) else modes
        if not models_to_plot:
            logger.error("No models specified for heatmap."); return None

        # 1. Get metric data
        df_metrics_raw: Optional[pd.DataFrame] = self.get_aggregated_metrics_dataframe(
            metric=metric, modes=_modes_list, models=models_to_plot, aggregation=metric_aggregation, # Use provided aggregation
            require_all_ts_ids=False, add_dataset_col=True
        )
        if df_metrics_raw is None or df_metrics_raw.empty:
            logger.error(f"Could not retrieve metric data for heatmap."); return None
        if 'dataset_name' not in df_metrics_raw.columns or df_metrics_raw['dataset_name'].nunique() == 0:
            logger.error("No 'dataset_name' column or no unique datasets in retrieved metrics."); return None

        # Determine initial_datasets_as_cols
        initial_datasets_as_cols: List[str]
        if datasets_as_cols is None:
            unique_datasets_from_data = sorted(df_metrics_raw['dataset_name'].unique().tolist())
            if not unique_datasets_from_data:
                logger.error("No datasets found for columns."); return None
            if len(unique_datasets_from_data) > max_auto_derived_datasets:
                logger.warning(f"Derived {len(unique_datasets_from_data)} dataset columns (>{max_auto_derived_datasets}). Heatmap may be wide.")
            initial_datasets_as_cols = unique_datasets_from_data
        else:
            initial_datasets_as_cols = sorted(list(set(datasets_as_cols)))
        if not initial_datasets_as_cols:
            logger.error("No dataset columns determined."); return None
        
        df_metrics_for_pivot: pd.DataFrame = df_metrics_raw[df_metrics_raw['dataset_name'].isin(initial_datasets_as_cols)].copy()
        if df_metrics_for_pivot.empty:
            logger.error(f"No metric data for initial columns: {initial_datasets_as_cols}."); return None

        # 2. Pivot table (raw metric values)
        try:
            # Use `metric_aggregation` for `aggfunc`
            pivot_df_initial: pd.DataFrame = df_metrics_for_pivot.pivot_table(
                index='model', columns='dataset_name', values=metric, aggfunc=metric_aggregation
            )
        except Exception as e:
            logger.error(f"Error pivoting data: {e}."); return None
        
        # Reindex to ensure all models_to_plot and initial_datasets_as_cols are present, filling NaNs where data is missing
        reindexed_pivot_df: pd.DataFrame = pivot_df_initial.reindex(index=models_to_plot, columns=initial_datasets_as_cols)

        # Determine final models and axes based on `keep_all_specified_models`
        final_models_for_rows: List[str]
        final_datasets_for_cols: List[str]
        raw_metric_pivot_df: pd.DataFrame

        if keep_all_specified_models:
            logger.info("Strategy: Keep models, drop incomplete dataset columns.")
            # Drop columns (datasets) that have any NaN for the specified models
            raw_metric_pivot_df = reindexed_pivot_df.dropna(axis=1, how='any')
            final_datasets_for_cols = raw_metric_pivot_df.columns.tolist()
            
            # Now, filter models: only keep models that have data in *at least one* of the *remaining* datasets
            # And also re-filter by models_to_plot in case some models were in original list but not in reindexed
            temp_df_models_filtered = raw_metric_pivot_df.dropna(axis=0, how='all')
            final_models_for_rows = [model for model in models_to_plot if model in temp_df_models_filtered.index]

            if final_models_for_rows and final_datasets_for_cols:
                raw_metric_pivot_df = raw_metric_pivot_df.loc[final_models_for_rows, final_datasets_for_cols]
            else:
                raw_metric_pivot_df = pd.DataFrame() # Empty if no common data
            
            removed_cols = set(initial_datasets_as_cols) - set(final_datasets_for_cols)
            if removed_cols: logger.warning(f"Dropped dataset columns due to missing data: {list(removed_cols)}")
            removed_rows = set(models_to_plot) - set(final_models_for_rows)
            if removed_rows: logger.warning(f"Dropped models (despite keep_all_specified_models=True) lacking data on common dataset axes: {list(removed_rows)}")
        else:
            logger.info("Strategy: Keep dataset columns, drop incomplete models.")
            # Drop rows (models) that have any NaN in the specified datasets
            raw_metric_pivot_df = reindexed_pivot_df.dropna(axis=0, how='any')
            final_models_for_rows = raw_metric_pivot_df.index.tolist()
            final_datasets_for_cols = initial_datasets_as_cols # All initial datasets are kept as columns
            removed_rows = set(models_to_plot) - set(final_models_for_rows)
            if removed_rows: logger.warning(f"Dropped models due to missing data: {list(removed_rows)}")

        if not final_models_for_rows or raw_metric_pivot_df.empty:
            logger.error("No models remaining for heatmap. Returning None."); return None
        if not final_datasets_for_cols:
            logger.error("No dataset columns remaining for heatmap. Returning None."); return None
        
        # 3. Normalize data per column for coloring (1=best, 0=worst in column)
        normalized_for_heatmap_df = pd.DataFrame(index=final_models_for_rows, columns=final_datasets_for_cols, dtype=float)
        for dataset_col in final_datasets_for_cols:
            col_data = raw_metric_pivot_df[dataset_col].astype(float)
            min_val, max_val = col_data.min(), col_data.max()
            
            if pd.isna(min_val) or min_val == max_val: 
                # If all values are NaN or identical, assign a neutral color (0.5) to non-NaN entries
                normalized_for_heatmap_df.loc[col_data.notna(), dataset_col] = 0.5 
            else:
                if lower_metric_is_better:
                    normalized_scores = (max_val - col_data) / (max_val - min_val)
                else:
                    normalized_scores = (col_data - min_val) / (max_val - min_val)
                normalized_for_heatmap_df.loc[:, dataset_col] = normalized_scores
        normalized_for_heatmap_df.fillna(0.5, inplace=True) # Fill any remaining NaNs (e.g., from original missing data) with neutral color

        if fig_size is None:
            fig_width = max(10, len(final_datasets_for_cols) * 0.9)
            fig_height = max(8, len(final_models_for_rows) * 0.6)
            fig_size = (fig_width, fig_height)
        fig, ax = plt.subplots(figsize=fig_size, dpi=200)

        # Define colorbar settings
        cbar_ticks = [0, 1]
        cbar_tick_labels = ["Worst", "Best"]
        
        cbar_arguments = {
            'label': f"Normalized {metric.upper()} per Dataset",
            'ticks': cbar_ticks,
            'format': plt.FuncFormatter(lambda x, p: cbar_tick_labels[int(x)]),
            'orientation': 'vertical'
        } 

        sns.heatmap(
            normalized_for_heatmap_df,
            annot=raw_metric_pivot_df.applymap(lambda x: f"{x:{annotation_fmt}}" if pd.notna(x) else ""),
            fmt="s",
            cmap='Blues', # Using Blues cmap, where 0 is white (worst) and 1 is dark blue (best)
            linewidths=0.5,
            linecolor='lightgray',
            ax=ax,
            cbar=True,
            cbar_kws=cbar_arguments,
            annot_kws={"size": 8}
        )

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Model")

        title_str = title if title else f"Model Performance Heatmap - Metric: {metric.upper()}"
        if not title:
            if len(_modes_list) == 1: title_str += f" (Mode: {_modes_list[0]})"
            elif len(_modes_list) > 1: title_str += f" (Modes: {', '.join(_modes_list)})"
        ax.set_title(title_str)
        
        plt.tight_layout(rect=[0, 0, 0.95, 1])

        if save_plot:
            save_dir = Path(save_path_base); save_dir.mkdir(parents=True, exist_ok=True)
            clean_metric = "".join(c if c.isalnum() else "_" for c in metric)
            if filename is None:
                filename = f"heatmap_metric_{clean_metric}_{'_'.join(_modes_list)}.png"
            try: fig.savefig(save_dir / filename, bbox_inches='tight', dpi=200)
            except Exception as e: logger.error(f"Error saving heatmap: {e}")
            else: logger.info(f"Heatmap saved to {save_dir / filename}")

        if show_plot: plt.show()
        plt.close(fig)
        return fig, ax

    def plot_models_radar_comparison(
        self,
        metric: str,
        modes: Union[str, List[str]],
        models_to_plot: List[str],
        datasets_as_axes: Optional[List[str]] = None,
        metric_aggregation: str = "mean",
        lower_metric_is_better: bool = True,
        max_auto_derived_axes: int = 15,
        keep_all_specified_models: bool = False,
        use_absolute_value_scaling: bool = True,
        save_plot: bool = False,
        save_path_base: Optional[Path] = PLOTS_DIR,
        filename: Optional[str] = None,
        show_plot: bool = True,
        title: Optional[str] = None,
        fig_size: Optional[Tuple[int, int]] = None
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Generates a radar plot comparing models' performance across different datasets (axes).
        Scaling can be absolute (value-based min-max per axis) or relative (rank-based per axis).

        Parameters
        ----------
        metric : str
            The performance metric (e.g., "mase").
        modes : Union[str, List[str]]
            Mode(s) to filter metric data.
        models_to_plot : List[str]
            List of model names to compare (each will be a line on the radar plot).
        datasets_as_axes : Optional[List[str]], default=None
            Optional list of dataset names to use as radar axes. If None, inferred from data.
        metric_aggregation : str, default="mean"
            Aggregation ("mean" or "median") for metric values if multiple `ts_id`s per model-dataset-mode.
        lower_metric_is_better : bool, default=True
            True if a lower metric value is better.
        max_auto_derived_axes : int, default=15
            Maximum number of dataset axes to auto-derive before a warning is logged.
        keep_all_specified_models : bool, default=False
            Strategy for handling missing data. If True, all `models_to_plot` are kept,
            and only dataset axes with complete data for these models are shown.
            If False, all dataset axes with any data are kept, and models with
            missing data in those axes are dropped.
        use_absolute_value_scaling : bool, default=True
            If True, scales values on each axis from 0 (worst) to 1 (best) based on
            the min/max metric values for that axis. If False, scales based on
            model ranks on each axis (Friedman ranks, where 1 is best, N is worst).
        save_plot : bool, default=False
            Whether to save the plot.
        save_path_base : Optional[Path], default=PLOTS_DIR
            Base directory for saving plots.
        filename : Optional[str], default=None
            Custom filename for the saved plot.
        show_plot : bool, default=True
            Whether to display the plot.
        title : Optional[str], default=None
            Optional custom title.
        fig_size : Optional[Tuple[int, int]], default=None
            Custom figure size (width, height) in inches.

        Returns
        -------
        Optional[Tuple[plt.Figure, plt.Axes]]
            Matplotlib Figure and Axes objects, or None if plotting fails.
        """
        _modes_list = [modes] if isinstance(modes, str) else modes
        if not models_to_plot:
            logger.error("No models specified for radar plot."); return None

        # 1. Get metric data
        df_metrics_raw: Optional[pd.DataFrame] = self.get_aggregated_metrics_dataframe(
            metric=metric, modes=_modes_list, models=models_to_plot, aggregation=metric_aggregation,
            require_all_ts_ids=False, add_dataset_col=True
        )
        if df_metrics_raw is None or df_metrics_raw.empty:
            logger.error(f"Could not retrieve metric data for radar plot."); return None
        if 'dataset_name' not in df_metrics_raw.columns or df_metrics_raw['dataset_name'].nunique() == 0:
            logger.error("No 'dataset_name' column or no unique datasets in retrieved metrics."); return None

        # Determine initial_datasets_as_axes
        initial_datasets_as_axes: List[str]
        if datasets_as_axes is None:
            unique_datasets_from_data = sorted(df_metrics_raw['dataset_name'].unique().tolist())
            if not unique_datasets_from_data:
                logger.error("No datasets found for axes."); return None
            if len(unique_datasets_from_data) > max_auto_derived_axes:
                logger.warning(f"Derived {len(unique_datasets_from_data)} axes (>{max_auto_derived_axes}). Plot may be cluttered.")
            initial_datasets_as_axes = unique_datasets_from_data
        else:
            initial_datasets_as_axes = sorted(list(set(datasets_as_axes)))
        if not initial_datasets_as_axes:
            logger.error("No dataset axes determined."); return None
        
        df_metrics_for_pivot: pd.DataFrame = df_metrics_raw[df_metrics_raw['dataset_name'].isin(initial_datasets_as_axes)].copy()
        if df_metrics_for_pivot.empty:
            logger.error(f"No metric data for initial axes: {initial_datasets_as_axes}."); return None

        # 2. Pivot table
        try:
            pivot_df_initial: pd.DataFrame = df_metrics_for_pivot.pivot_table(
                index='model', columns='dataset_name', values=metric, aggfunc=metric_aggregation
            )
        except Exception as e:
            logger.error(f"Error pivoting data: {e}."); return None
        
        reindexed_pivot_df: pd.DataFrame = pivot_df_initial.reindex(index=models_to_plot, columns=initial_datasets_as_axes)

        # Determine final models and axes based on `keep_all_specified_models`
        final_models_to_plot: List[str]
        final_dataset_axes: List[str]
        final_pivot_df: pd.DataFrame

        if keep_all_specified_models:
            logger.info("Strategy: Keep models, drop incomplete dataset axes.")
            final_pivot_df = reindexed_pivot_df.dropna(axis=1, how='any') # Keep axes (cols) that are complete
            final_dataset_axes = final_pivot_df.columns.tolist()
            
            temp_df_models_filtered = final_pivot_df.dropna(axis=0, how='all') # Keep models (rows) that have any data
            final_models_to_plot = [model for model in models_to_plot if model in temp_df_models_filtered.index] # Ensure from original list
            
            if final_models_to_plot and final_dataset_axes:
                final_pivot_df = final_pivot_df.loc[final_models_to_plot, final_dataset_axes]
            else:
                final_pivot_df = pd.DataFrame() # Empty if no common data
            
            removed_datasets = set(initial_datasets_as_axes) - set(final_dataset_axes)
            if removed_datasets: logger.warning(f"Dropped dataset axes due to missing data: {list(removed_datasets)}")
            removed_models = set(models_to_plot) - set(final_models_to_plot)
            if removed_models: logger.warning(f"Dropped models (despite keep_all_specified_models=True) lacking data on common axes: {list(removed_models)}")
        else:
            logger.info("Strategy: Keep dataset axes, drop incomplete models.")
            final_pivot_df = reindexed_pivot_df.dropna(axis=0, how='any') # Keep models (rows) that are complete
            final_models_to_plot = final_pivot_df.index.tolist()
            final_dataset_axes = initial_datasets_as_axes # All initial datasets are kept as axes
            removed_models = set(models_to_plot) - set(final_models_to_plot)
            if removed_models: logger.warning(f"Dropped models due to missing data: {list(removed_models)}")

        if not final_models_to_plot or final_pivot_df.empty:
            logger.error("No models remaining. Cannot create radar plot."); return None
        if len(final_dataset_axes) < 3:
            logger.error(f"Less than 3 axes ({len(final_dataset_axes)}) remaining. Cannot create radar plot."); return None
        
        # 3. Normalize data
        normalized_df: pd.DataFrame = pd.DataFrame(index=final_models_to_plot, columns=final_dataset_axes, dtype=float)

        for dataset_axis in final_dataset_axes:
            col_data: pd.Series = final_pivot_df.loc[final_models_to_plot, dataset_axis].astype(float)

            if use_absolute_value_scaling:
                min_val, max_val = col_data.min(), col_data.max()
                if pd.isna(min_val) or min_val == max_val:
                    normalized_df.loc[col_data.notna(), dataset_axis] = 0.5
                else:
                    if lower_metric_is_better:
                        normalized_df.loc[:, dataset_axis] = (max_val - col_data) / (max_val - min_val)
                    else:
                        normalized_df.loc[:, dataset_axis] = (col_data - min_val) / (max_val - min_val)
            else: # Rank-based scaling
                num_models_for_ranking: int = col_data.dropna().count()
                
                # Rank data: smallest value gets rank 1 if ascending=True
                ranks: pd.Series = col_data.rank(method='average', ascending=lower_metric_is_better, na_option='keep')
                
                if num_models_for_ranking > 1:
                    # Scale ranks: rank 1 (best) maps to 1.0, rank N (worst) maps to 0.0
                    scaled_ranks: pd.Series = (num_models_for_ranking - ranks) / (num_models_for_ranking - 1)
                    normalized_df.loc[:, dataset_axis] = scaled_ranks
                else: # Only one model has data for this axis, or all others are NaN
                    normalized_df.loc[col_data.notna(), dataset_axis] = 0.5 # Neutral score for the single valid model

        normalized_df.fillna(0.5, inplace=True) # Fill any NaNs (e.g., from original missing data) with neutral 0.5

        # 4. Plotting setup
        num_vars: int = len(final_dataset_axes)
        angles: np.ndarray = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        closed_angles: np.ndarray = np.concatenate((angles, [angles[0]])) # Close the plot circle
        
        if fig_size is None:
            fig_size = (10, 10)
        fig, ax = plt.subplots(figsize=fig_size, dpi=200, subplot_kw=dict(polar=True))
        
        def wrap_axis_labels(labels: List[str], max_len: int = 20, max_lines: int = 2) -> List[str]:
            """Helper to wrap long axis labels for better readability."""
            wrapped_labels = []
            for label in labels:
                if len(label) > max_len:
                    parts = label.replace('_', ' ').split(' ')
                    lines: List[str] = [""]
                    line_idx: int = 0
                    for part in parts:
                        if len(lines[line_idx]) + len(part) + 1 <= max_len:
                            lines[line_idx] += f"{part} "
                        elif line_idx + 1 < max_lines:
                            lines.append(f"{part} "); line_idx += 1
                        else:
                            lines[line_idx] += f"{part} "; break 
                    wrapped_labels.append("\n".join(l.strip() for l in lines))
                else:
                    wrapped_labels.append(label.replace('_', ' '))
            return wrapped_labels
        
        axis_labels_display: List[str] = wrap_axis_labels(final_dataset_axes)
        ax.set_xticks(angles)
        ax.set_xticklabels(axis_labels_display)
        
        # Set y-axis (radial) ticks to represent the normalized scale from 0 to 1
        ax.set_yticks(np.arange(0, 1.1, 0.25))
        ax.set_yticklabels([f"{tick:.2f}" for tick in np.arange(0, 1.1, 0.25)])
        ax.set_ylim(0, 1.05)


        if len(final_models_to_plot) <= 10: model_colors = sns.color_palette("tab10", n_colors=len(final_models_to_plot))
        elif len(final_models_to_plot) <= 20: model_colors = sns.color_palette("tab20", n_colors=len(final_models_to_plot))
        else: model_colors = sns.color_palette("husl", n_colors=len(final_models_to_plot))

        for model_idx, model_name in enumerate(final_models_to_plot):
            values: List[float] = normalized_df.loc[model_name, final_dataset_axes].tolist()
            closed_values: List[float] = values + [values[0]]
            line_color = model_colors[model_idx]
            ax.plot(closed_angles, closed_values, label=model_name, linewidth=2.5, linestyle='solid', color=line_color, zorder=len(final_models_to_plot) - model_idx)

        title_str = title if title else f"Model Comparison - Metric: {metric.upper()}"
        if not title: 
            if len(_modes_list) == 1: title_str += f" (Mode: {_modes_list[0]})"
            elif len(_modes_list) > 1: title_str += f" (Modes: {', '.join(_modes_list)})"
        scaling_type = "Value-Based Scaling" if use_absolute_value_scaling else "Rank-Based Scaling"
        title_str += f"\n({scaling_type})"
        ax.set_title(title_str, size=15, y=1.12, weight='bold') # Adjusted title size
        
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.20), ncol=min(3, len(final_models_to_plot)), frameon=True)
        fig.tight_layout(pad=2.5) # Adjusted padding

        if save_plot:
            save_dir = Path(save_path_base); save_dir.mkdir(parents=True, exist_ok=True)
            clean_metric = "".join(c if c.isalnum() else "_" for c in metric)
            scaling_suffix = "abs_scale" if use_absolute_value_scaling else "rank_scale"
            if filename is None:
                filename = f"radar_lines_{clean_metric}_{'_'.join(_modes_list)}_{scaling_suffix}.png"
            try: fig.savefig(save_dir / filename, bbox_inches='tight', dpi=150)
            except Exception as e: logger.error(f"Error saving radar plot: {e}")
            else: logger.info(f"Radar plot saved to {save_dir / filename}")

        if show_plot: plt.show()
        plt.close(fig)
        return fig, ax

    def plot_model_comparison_per_dataset(
        self,
        metric: str,
        modes: Union[str, List[str]],
        models_to_compare: Union[str, List[str]],
        plot_aggregation: str = "mean",
        save_plot: bool = False,
        save_path_base: Optional[Path] = PLOTS_DIR,
        filename: Optional[str] = None,
        show_plot: bool = True,
        title: Optional[str] = None,
        fig_size: Optional[Tuple[int, int]] = None,
        errorbar_type: str = 'ci',
        sort_ascending: bool = True,
        add_legend_for_model_categories: bool = True
    ) -> List[Optional[Tuple[plt.Figure, plt.Axes]]]:
        """
        Generates a horizontal bar plot for each dataset, comparing specified models
        based on an aggregated metric.

        The input `metric` values are expected to be at the `ts_id` level for each dataset.
        This function then uses `plot_aggregation` (e.g., "mean") to determine bar heights
        (via seaborn's estimator) and sorting order within each dataset's plot.

        Parameters
        ----------
        metric : str
            The metric to plot (e.g., "mase").
        modes : Union[str, List[str]]
            A single mode name (str) or a list of mode names to filter data.
        models_to_compare : Union[str, List[str]]
            A list of model names to be shown on the y-axis of the bar plots.
        plot_aggregation : str, default="mean"
            Aggregation method ("mean" or "median") for seaborn.barplot estimator and sorting.
        save_plot : bool, default=False
            Whether to save the plots.
        save_path_base : Optional[Path], default=PLOTS_DIR
            Base directory to save plots. A subdirectory "per_dataset_model_comparisons" will be created here.
        filename : Optional[str], default=None
            Custom filename for the saved plot. If None, a default filename is generated.
        show_plot : bool, default=True
            Whether to display plots. Useful to set to False if many plots are generated.
        title : Optional[str], default=None
            Optional custom title for individual plots.
        fig_size : Optional[Tuple[int, int]], default=None
            Custom figure size (width, height) in inches for each plot.
        errorbar_type : str, default='ci'
            Type of error bar for seaborn.barplot (e.g., 'ci', 'sd').
        sort_ascending : bool, default=True
            If True, sorts models by the aggregated metric in ascending order within each plot.
        add_legend_for_model_categories : bool, default=True
            If True, adds a legend for model categories.

        Returns
        -------
        List[Optional[Tuple[plt.Figure, plt.Axes]]]
            A list of (figure, axes) tuples for each generated plot, or an empty list if errors occur.
        """
        _models_to_compare_list = [models_to_compare] if isinstance(models_to_compare, str) else models_to_compare
        _modes_list = [modes] if isinstance(modes, str) else modes

        df_all_datasets_metrics: Optional[pd.DataFrame] = self.get_aggregated_metrics_dataframe(
            metric=metric,
            modes=_modes_list,
            models=_models_to_compare_list,
            aggregation="mean", # This internally aggregates to ts_id level
            require_all_ts_ids=False,
            add_dataset_col=True,
            add_short_long_term_col=False
        )

        if df_all_datasets_metrics is None or df_all_datasets_metrics.empty:
            logger.error(f"Failed to retrieve data for metric '{metric}'. Cannot generate model comparison per dataset.")
            return []

        if 'dataset_name' not in df_all_datasets_metrics.columns:
            logger.error("'dataset_name' column not found in the retrieved data. Cannot proceed.")
            return []

        unique_datasets: List[str] = sorted(df_all_datasets_metrics['dataset_name'].unique())
        logger.info(f"Found {len(unique_datasets)} unique datasets to plot for: {unique_datasets}")

        plotted_figures_axes: List[Optional[Tuple[plt.Figure, plt.Axes]]] = []
        
        per_dataset_save_path: Path = save_path_base / "per_dataset_model_comparisons"
        if save_plot:
            per_dataset_save_path.mkdir(parents=True, exist_ok=True)

        for dataset_name in unique_datasets:
            df_dataset_current: pd.DataFrame = df_all_datasets_metrics[
                (df_all_datasets_metrics['dataset_name'] == dataset_name) &
                (df_all_datasets_metrics['model'].isin(_models_to_compare_list))
            ].copy()

            if df_dataset_current.empty:
                logger.warning(f"No data for dataset '{dataset_name}' with the specified models/modes. Skipping plot.")
                continue

            metric_col: str = metric
            df_plot_data: pd.DataFrame = df_dataset_current.dropna(subset=[metric_col]).copy()

            if df_plot_data.empty:
                logger.warning(f"After dropping rows with NaN in '{metric_col}' for dataset '{dataset_name}', DataFrame is empty. Skipping plot.")
                continue

            y_axis_col: str = 'model'
            hue_col: Optional[str] = None
            palette_to_use: Optional[Dict[str, str]] = None
            plot_actual_legend: bool = False

            if add_legend_for_model_categories and self.model_categories:
                df_plot_data['model_category'] = df_plot_data['model'].map(self.model_categories).fillna('Other')
                hue_col = 'model_category'
                plot_actual_legend = True
                if self.category_colors:
                    actual_categories_in_data = df_plot_data['model_category'].unique()
                    palette_to_use = {cat: self.category_colors.get(cat, '#808080') for cat in actual_categories_in_data}
            
            estimator_func_for_plot: Union[str, callable]
            if plot_aggregation.lower() == "mean":
                estimator_func_for_plot = np.mean
                order_determining_values = df_plot_data.groupby(y_axis_col, observed=True)[metric_col].mean()
            elif plot_aggregation.lower() == "median":
                estimator_func_for_plot = np.median
                order_determining_values = df_plot_data.groupby(y_axis_col, observed=True)[metric_col].median()
            else:
                logger.warning(f"Invalid plot_aggregation '{plot_aggregation}'. Defaulting to 'mean'.")
                estimator_func_for_plot = np.mean
                order_determining_values = df_plot_data.groupby(y_axis_col, observed=True)[metric_col].mean()
            
            valid_models_for_plot: List[str] = order_determining_values.index.tolist()
            df_plot_data_final: pd.DataFrame = df_plot_data[df_plot_data[y_axis_col].isin(valid_models_for_plot)].copy()
            
            if df_plot_data_final.empty:
                logger.warning(f"No models with valid aggregated data for dataset '{dataset_name}' to plot. Skipping.")
                continue

            sorted_model_categories: List[str] = order_determining_values.sort_values(ascending=sort_ascending).index.tolist()
            
            if fig_size is None:
                fig_size = (14, max(6, len(sorted_model_categories) * 0.5))
            fig, ax = plt.subplots(figsize=fig_size, dpi=200)
            
            sns.barplot(
                data=df_plot_data_final,
                y=y_axis_col,
                x=metric_col,
                hue=hue_col,
                ax=ax,
                estimator=estimator_func_for_plot,
                errorbar=errorbar_type,
                capsize=0.1,
                palette=palette_to_use,
                orient='h',
                order=sorted_model_categories
            )

            current_title_base: str = f"{plot_aggregation.capitalize()} {metric_col.upper()} for Dataset: {dataset_name}"
            if len(_modes_list) == 1:
                current_title = f"{current_title_base}\n(Mode: {_modes_list[0]})"
            else:
                current_title = f"{current_title_base}\n(Modes: {', '.join(_modes_list)})"
            
            if title:
                current_title = title

            ax.set_title(current_title)
            ax.set_xlabel(f'{plot_aggregation.capitalize()} {metric_col.upper()}')
            ax.set_ylabel(y_axis_col.replace('_', ' ').capitalize())

            if plot_actual_legend and hue_col and ax.get_legend() is not None:
                ax.legend(title=hue_col.replace('_', ' ').capitalize(), loc='upper right')
                plt.tight_layout()
            elif ax.get_legend() is not None:
                ax.get_legend().remove()
                plt.tight_layout()
            else:
                plt.tight_layout()

            clean_dataset_name: str = "".join(c if c.isalnum() else "_" for c in dataset_name)
            if filename is None:
                filename_models_str = "_".join(_models_to_compare_list) if len(_models_to_compare_list) <= 3 else "selected_models"
                filename_modes_str = "_".join(_modes_list) if len(_modes_list) <= 3 else "selected_modes"
                filename = f"barplot_{metric_col}_dataset_{clean_dataset_name}_{plot_aggregation}_{filename_models_str}_{filename_modes_str}.png"
                filename = filename.replace(" ", "_").replace("+", "plus").lower() # Sanitize filename
            
            self._plot_save_and_show(fig, per_dataset_save_path, filename, show_plot=show_plot, save_plot=save_plot)
            plotted_figures_axes.append((fig, ax))

        return plotted_figures_axes

    def plot_timing_analysis_single_model(
        self,
        model: str,
        modes: Union[str, List[str]] = PRETTY_MODES,
        save_path: Optional[Path] = PLOTS_DIR,
        save_plot: bool = False,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        fig_size: Optional[Tuple[int, int]] = None,
        show_plot: bool = True
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Generates a scatter plot of forecast time vs. time series dimension for a single model,
        with time series length categorized and used as hue.

        Parameters
        ----------
        model : str
            The single model name to analyze.
        modes : Union[str, List[str]], default=PRETTY_MODES
            Single user-friendly mode name (str) or list of mode names to include for this model.
        save_path : Optional[Path], default=PLOTS_DIR
            Path to save the generated plot image.
        save_plot : bool, default=False
            If True, saves the plot to a file.
        filename : Optional[str], default=None
            Custom filename for the saved plot. If None, a default filename is generated.
        title : Optional[str], default=None
            Custom title for the scatter plot. If None, a default title is generated.
        fig_size : Optional[Tuple[int, int]], default=None
            Custom figure size (width, height) in inches. If None, (10, 8) is used.
        show_plot : bool, default=True
            Whether to display the plot.

        Returns
        -------
        Optional[Tuple[plt.Figure, plt.Axes]]
            A tuple containing the matplotlib Figure and Axes objects, or None if plotting fails.
        """
        _models_list = [model]
        _modes_list = [modes] if isinstance(modes, str) else modes

        df_timing: Optional[pd.DataFrame] = self.get_timing_dataframe(
            models=_models_list,
            modes=_modes_list,
            require_all_samples=False
        )

        if df_timing is None or df_timing.empty:
            logger.error(f"Failed to retrieve timing data for model '{model}' in modes '{_modes_list}'. Cannot generate plot.")
            return None

        length_bins: List[Union[int, float]] = [0, 100, 1000, 10000, np.inf]
        length_labels: List[str] = ['<= 100', '101-1000', '1001-10000', '> 10000']
        
        df_timing['length_category'] = pd.cut(
            df_timing['length'],
            bins=length_bins,
            labels=length_labels,
            right=True,
            include_lowest=True,
            ordered=True
        )
        
        length_palette_colors = sns.color_palette("viridis", n_colors=len(length_labels))
        custom_length_colors = {
            label: length_palette_colors[i] for i, label in enumerate(length_labels)
        }

        if fig_size is None:
            fig_size = (10, 8)
        fig, ax = plt.subplots(figsize=fig_size, dpi=200)

        sns.scatterplot(
            data=df_timing,
            x='dimension',
            y='forecast_time_seconds',
            hue='length_category',
            palette=custom_length_colors,
            ax=ax,
            s=100,
            alpha=0.7
        )

        if title is None:
            title = f"Forecast Time vs. Dimension for Model: {model} (Modes: {', '.join(_modes_list)})"
        ax.set_title(title)
        ax.set_xlabel("Time Series Dimension")
        ax.set_ylabel("Forecast Time (seconds)")
        ax.grid(True, linestyle=':', alpha=0.6)

        # Apply log scale if values are positive and span a wide range
        if (df_timing['forecast_time_seconds'] > 0).all():
            ax.set_yscale('log')
            ax.set_ylabel("Forecast Time (seconds, log scale)")
        if (df_timing['dimension'] > 0).all():
            ax.set_xscale('log')
            ax.set_xlabel("Time Series Dimension (log scale)")

        if ax.get_legend() is not None:
            ax.legend(title='Time Series Length', loc='upper left')

        plt.tight_layout()
        if filename is None:
            filename_modes_str = "_".join(_modes_list) if len(_modes_list) <= 3 else "selected_modes"
            filename = f"timing_scatter_{model.replace(' ', '_')}_{filename_modes_str}.png"
            filename = filename.replace("+", "plus").lower() # Sanitize filename
        self._plot_save_and_show(fig, save_path, filename, show_plot, save_plot)
        return fig, ax
    
    def plot_timing_barplot(
        self,
        models: Union[str, List[str]],
        modes: Union[str, List[str]] = COV_MODES,
        aggregation: str = "mean",
        save_path: Optional[Path] = PLOTS_DIR,
        save_plot: bool = False,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        fig_size: Optional[Tuple[int, int]] = None,
        add_legend: bool = True,
        errorbar_type: str = 'ci',
        sort_ascending: bool = True
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Generates a bar plot comparing the average (or median) forecast timing across different models.

        Parameters
        ----------
        models : Union[str, List[str]]
            Single user-friendly model name (str) or list of model names to compare.
        modes : Union[str, List[str]], default=COV_MODES
            Single user-friendly mode name (str) or list of mode names to filter the timing data.
        aggregation : str, default="mean"
            The aggregation method to use for timing ("mean" or "median").
        save_path : Optional[Path], default=PLOTS_DIR
            Path to save the generated plot image.
        save_plot : bool, default=False
            If True, saves the plot to a file.
        filename : Optional[str], default=None
            Custom filename for the saved plot.
        title : Optional[str], default=None
            Custom title for the bar plot.
        fig_size : Optional[Tuple[int, int]], default=None
            Custom figure size (width, height) in inches.
        add_legend : bool, default=True
            If True, adds a legend based on model categories.
        errorbar_type : str, default='ci'
            Type of error bar to display ('ci', 'sd', etc.).
        sort_ascending : bool, default=True
            If True, sort bars in ascending order of time; otherwise descending.

        Returns
        -------
        Optional[Tuple[plt.Figure, plt.Axes]]
            A tuple containing the matplotlib Figure and Axes objects, or None if plotting fails.
        """
        _models_list = [models] if isinstance(models, str) else models
        _modes_list = [modes] if isinstance(modes, str) else modes

        df_timing_raw: Optional[pd.DataFrame] = self.get_timing_dataframe(
            models=_models_list,
            modes=_modes_list,
            require_all_samples=False # Changed to False, as `require_all_samples=True` severely limits data for barplot
        )

        if df_timing_raw is None or df_timing_raw.empty:
            logger.error("Failed to retrieve timing data for bar plot. Cannot generate plot.")
            return None

        df_timing_raw['forecast_time_seconds'] = pd.to_numeric(df_timing_raw['forecast_time_seconds'], errors='coerce')
        df_timing: pd.DataFrame = df_timing_raw.dropna(subset=['forecast_time_seconds']).copy()

        if df_timing.empty:
            logger.warning(f"After dropping rows with NaN in 'forecast_time_seconds', the DataFrame is empty. Cannot generate bar plot.")
            return None

        y_label_prefix: str
        if aggregation == "mean":
            df_aggregated_timing = df_timing.groupby('model', observed=True)['forecast_time_seconds'].mean().reset_index()
            y_label_prefix = "Mean"
        elif aggregation == "median":
            df_aggregated_timing = df_timing.groupby('model', observed=True)['forecast_time_seconds'].median().reset_index()
            y_label_prefix = "Median"
        else:
            logger.error(f"Invalid aggregation method '{aggregation}'. Use 'mean' or 'median'.")
            return None

        df_aggregated_timing = df_aggregated_timing.sort_values(
            by='forecast_time_seconds', ascending=sort_ascending
        )
        sorted_models: List[str] = df_aggregated_timing['model'].tolist()

        hue_col: Optional[str] = None
        palette_to_use: Optional[Dict[str, str]] = None
        if add_legend and self.model_categories and len(_models_list) > 1:
            df_timing['model_category'] = df_timing['model'].map(self.model_categories).fillna('Other')
            hue_col = 'model_category'
            actual_categories = df_timing['model_category'].unique()
            palette_to_use = {cat: self.category_colors.get(cat, '#808080') for cat in actual_categories}
        
        if fig_size is None:
            fig_size = (14, 8)
        fig, ax = plt.subplots(figsize=fig_size, dpi=200)
        
        sns.barplot(
            data=df_timing,
            x='model',
            y='forecast_time_seconds',
            hue=hue_col,
            palette=palette_to_use,
            order=sorted_models,
            ax=ax,
            errorbar=errorbar_type,
            capsize=0.1
        )

        if title is None:
            title = f"{y_label_prefix} Forecast Time by Model (Modes: {', '.join(_modes_list)})"
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.set_ylabel(f"{y_label_prefix} Forecast Time in Seconds")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        if hue_col and ax.get_legend() is not None:
            ax.legend(title='Model Category', loc='upper left')
            plt.tight_layout()
        elif ax.get_legend() is not None:
            ax.get_legend().remove()
            plt.tight_layout()
        else:
            plt.tight_layout()
        
        if filename is None:
            filename_models_str = "_".join(_models_list) if len(_models_list) <= 3 else "selected_models"
            filename_modes_str = "_".join(_modes_list) if len(_modes_list) <= 3 else "selected_modes"
            filename = f"timing_barplot_{aggregation}_{filename_models_str}_{filename_modes_str}.png"
            filename = filename.replace(" ", "_").replace("+", "plus").lower() # Sanitize filename

        self._plot_save_and_show(fig, save_path, filename, show_plot=True, save_plot=save_plot)
        return fig, ax