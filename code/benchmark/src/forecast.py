import csv
import json
import logging
import numpy as np
import uuid
import ast
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

from src.data.data_loader import COV_MODES, DataLoader, TSInfo
from src.models.cpu_models.baseline_models import BaselineModels
from src.models.cpu_models.other_models import OtherModels
from src.models.gpu_models.fm_api import FoundationModelAPIClient
from src.models.gpu_models.timegpt import TimeGPT
from src.models.schemas import ModelInput, ModelOutput

# --- Configuration Paths ---
DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
FORECAST_DIR: Path = DATA_DIR / "forecasts"
FORECASTS_META_PATH: Path = FORECAST_DIR / "forecasts_meta.csv"
FORECASTS_DATA_PATH: Path = FORECAST_DIR / "forecasts_data.csv"
FEATURE_IMPORTANCE_PATH: Path = FORECAST_DIR / "feature_importance.jsonl"
LOG_DIR: Path = Path(__file__).resolve().parent.parent / "logs"
LOG_PATH: Path = LOG_DIR / "forecast.log"

# --- Forecasting Process Constants ---
STORE_FREQUENCY: int = 30  # How often to save accumulated forecasts to CSV.
FAILURE_THRESHOLD_PERCENT: float = 0.1  # Percentage of failures to trigger dataset skip.
MIN_FAILURES_FOR_DATASET_SKIP: int = 3 # Minimum consecutive failures before considering dataset skip.
MAX_DISTINCT_DATASETS_FOR_MODEL_SKIP: int = 3 # Max distinct datasets failing before skipping a model setup.

# --- CSV Fieldnames ---
DATA_FIELDNAMES: List[str] = [
    "forecast_id",
    "ts_id",
    "step",
    "point_forecast",
    "lower_bound",
    "upper_bound",
]
META_FIELDNAMES: List[str] = [
    "model",
    "mode",
    "sample_id",
    "forecast_successful",
    "forecast_time_seconds",
    "error_log_id",
    "forecast_ids",
]

# --- Logging Configuration ---
LOGGER_NAME: str = "Forecast"
LOG_FORMAT: str = '%(asctime)s - %(levelname)s) - [%(log_id)s] - %(message)s'


# Set up logger.
logger: logging.Logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Ensure log directory exists before setting up file handler.
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(LOG_PATH)
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


class ModelSetup:
    """
    A simple data class to define the configuration for a specific model.

    Attributes
    ----------
    model_category : str
        The high-level category of the model (e.g., "baseline", "other", "timegpt", "fm_api").
    **kwargs
        Additional keyword arguments specific to the model, passed to its `forecast` method.
    """

    def __init__(self, model_category: str, **kwargs: Any) -> None:
        self.model_category = model_category
        for key, value in kwargs.items():
            setattr(self, key, value)


# --- Model Setup Configurations ---
# These define which specific models to run and with what parameters.
# The keys are internal names used in the Forecaster to identify a specific model configuration.
MODEL_SETUPS: Dict[str, ModelSetup] = {
    "all_baseline": ModelSetup("baseline", models=["AutoARIMA", "AutoETS", "AutoTheta"]),
    "arima": ModelSetup("baseline", models=["AutoARIMA"]),
    "gbm": ModelSetup("other", model="regressor", model_type="GBM"),
    "chronos_base": ModelSetup("other", model="chronos", model_type="bolt_base", regressor="LR"),
    # "chronos_small": ModelSetup("other", model="chronos", model_type="bolt_small", regressor="LR"),
    "chronos_gbm": ModelSetup("other", model="chronos", model_type="bolt_base", regressor="GBM"),
    "timegpt": ModelSetup("timegpt"),
    "timegpt_finetuned": ModelSetup("timegpt", finetune=True),
    "ttm": ModelSetup("fm_api", model="ttm", model_path="ibm-granite/granite-timeseries-ttm-r2"),
    "ttm_finetuned": ModelSetup("fm_api", model="ttm", model_path="ibm-granite/granite-timeseries-ttm-r2", kwargs={"fine_tune_perc": 1}),
    # "timesfm_v1": ModelSetup("fm_api", model="timesfm", model_path="google/timesfm-1.0-200m-pytorch", kwargs={"horizon_len": 256}),
    "timesfm_v2": ModelSetup("fm_api", model="timesfm", model_path="google/timesfm-2.0-500m-pytorch", kwargs={"horizon_len": 256}),
    "timesfm_post_xreg": ModelSetup("fm_api", model="timesfm", model_path="google/timesfm-2.0-500m-pytorch", kwargs={"horizon_len": 256, "xreg_mode": "timesfm + xreg"}),
    # "moirai_small": ModelSetup("fm_api", model="moirai", model_path="Salesforce/moirai-1.1-R-small"),
    "moirai_base": ModelSetup("fm_api", model="moirai", model_path="Salesforce/moirai-1.1-R-base"),
    # "moirai_batch2": ModelSetup("fm_api", model="moirai", model_path="Salesforce/moirai-1.1-R-base", kwargs={"batch_size": 2}),
    "moirai_moe": ModelSetup("fm_api", model="moirai", model_path="Salesforce/moirai-moe-1.0-R-base"),
    # "NHITS": ModelSetup("fm_api", model="NHITS"),
    # "TFT": ModelSetup("fm_api", model="TFT"),
    "TiDE": ModelSetup("fm_api", model="TiDE"),
    "NBEATSx": ModelSetup("fm_api", model="NBEATSx"),
}


class CSVWriter:
    """
    A utility class for writing data to CSV files.
    It supports writing single rows or multiple rows and ensures the header is written
    only when the file is created for the first time.
    """

    def __init__(self, file_path: Path, fieldnames: Optional[List[str]] = None) -> None:
        """
        Initializes the CSVWriter.

        Parameters
        ----------
        file_path : Path
            The full path to the CSV file. Parent directories will be created if they don't exist.
        fieldnames : Optional[List[str]], default=None
            A list of column names to be used as the CSV header. If the file does not
            exist, this header will be written. Required if the file doesn't exist.
        """
        self.file_path = file_path
        self.fieldnames = fieldnames

        # Ensure parent directory exists.
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.file_path.exists():
            if not self.fieldnames:
                raise ValueError(
                    f"Fieldnames must be provided to create a new CSV file: {self.file_path}"
                )
            with open(self.file_path, mode="w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
                logger.info(f"Created new CSV file with header: {self.file_path}")

    def write_row(self, row: Dict[str, Any]) -> None:
        """
        Writes a single row (dictionary) to the CSV file.

        Parameters
        ----------
        row : Dict[str, Any]
            A dictionary representing the row to write, where keys match the fieldnames.
        """
        with open(self.file_path, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(row)

    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        """
        Writes multiple rows (list of dictionaries) to the CSV file.

        Parameters
        ----------
        rows : List[Dict[str, Any]]
            A list of dictionaries, each representing a row to write.
        """
        with open(self.file_path, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerows(rows)


class Forecaster:
    """
    Manages the end-to-end forecasting pipeline, including data loading,
    model execution, and result storage. It incorporates intelligent
    skipping logic to efficiently resume interrupted runs and
    handle recurring model failures.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        meta_path: Path = FORECASTS_META_PATH,
        data_path: Path = FORECASTS_DATA_PATH,
        process: str = "",
        redo_failed_forecasts: bool = False,
    ) -> None:
        """
        Initializes the Forecaster.

        Parameters
        ----------
        data_loader : DataLoader
            An initialized DataLoader instance to fetch input data samples.
        meta_path : Path, default=FORECASTS_META_PATH
            Path to the CSV file storing forecast metadata (e.g., model used, success status).
        data_path : Path, default=FORECASTS_DATA_PATH
            Path to the CSV file storing detailed forecast data (point forecasts, intervals).
        process : str, default=""
            An identifier for the current forecasting process, used in log messages.
        redo_failed_forecasts : bool, default=False
            If True, previously failed forecasts recorded in the meta file will be re-attempted.
            If False, both successful and failed forecasts will be skipped.
        """
        self.dl = data_loader
        self.meta_path = meta_path
        self.data_path = data_path
        self.process = process
        self.redo_failed_forecasts = redo_failed_forecasts

        self.samples = self.dl.samples
        self.ts_ids = self.dl.ts_ids

        # Initialize model instances.
        self.models: Dict[str, Union[BaselineModels, OtherModels, TimeGPT, FoundationModelAPIClient]] = {
            "baseline": BaselineModels(),
            "other": OtherModels(),
            "timegpt": TimeGPT(),
            "fm_api": FoundationModelAPIClient(),
        }

        # Buffers to hold forecast results before writing to CSV.
        self.forecasts: List[ModelOutput] = []
        self.forecasts_info: List[Dict[str, Any]] = []

        # --- Skipping Logic State Variables ---
        # `completed_forecasts` stores (model_name, mode_name, sample_id) tuples that are
        # already processed (successfully or unsuccessfully, depending on `redo_failed_forecasts`).
        self.completed_forecasts: set[Tuple[str, str, int]] = set()

        # `failed_windows` implements Rule 1: Skip subsequent windows after first failure for
        # (model_setup_str, mode, dataset_name, ts_name, fc_horizon).
        self.failed_windows: set[Tuple[str, str, str, str, int]] = set()

        # `consecutive_dataset_failures` implements Rule 2 (part 1): Tracks consecutive failures
        # for a (model_setup_str, mode, dataset_name) combination.
        self.consecutive_dataset_failures: Dict[Tuple[str, str, str], int] = defaultdict(int)

        # `skipped_datasets` implements Rule 2 (part 2): Marks a dataset as fully skipped for
        # a (model_setup_str, mode) if its failure threshold is hit.
        self.skipped_datasets: set[Tuple[str, str, str]] = set()

        # `distinct_datasets_skipped_by_rule2` implements Rule 3 (part 1): Tracks distinct datasets
        # that were skipped by Rule 2 for a (model_setup_str, mode) combination.
        self.distinct_datasets_skipped_by_rule2: Dict[Tuple[str, str], set[str]] = defaultdict(set)

        # `skipped_model_setups` implements Rule 3 (part 2): Marks a model setup/mode combination
        # as entirely skipped if too many distinct datasets failed for it.
        self.skipped_model_setups: set[Tuple[str, str]] = set()
        
        # Load any existing completed forecasts from the meta file at initialization.
        self._load_completed_forecasts()

    def _load_completed_forecasts(self) -> None:
        """
        Loads previously completed forecast metadata from `self.meta_path` to populate
        `self.completed_forecasts` and determines the maximum existing `forecast_id`
        to ensure new IDs are unique and sequential.

        It respects `self.redo_failed_forecasts`: if True, only successfully
        completed forecasts are added to `self.completed_forecasts` to allow retries
        for failed ones.
        """
        self.forecast_idx: int = 0  # Initialize forecast index.
        max_forecast_idx_found_so_far: int = 0  # Track the highest existing forecast ID.

        # If the meta file does not exist, start fresh.
        if not self.meta_path.exists():
            logger.info(
                f"({self.process}) Meta file {self.meta_path} not found. Starting fresh. "
                "Forecast_idx set to 0.",
                extra={"log_id": ""},
            )
            return

        try:
            with open(self.meta_path, mode="r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                # Define required fields for a valid meta entry.
                required_fields: List[str] = [
                    "model", "mode", "sample_id", "forecast_successful", "forecast_ids"
                ]

                # Check if all required columns exist in the CSV header.
                if not all(field in reader.fieldnames for field in required_fields):
                    logger.warning(
                        f"({self.process}) Meta file {self.meta_path} is missing required columns "
                        f"(need at least {required_fields}). Cannot reliably load completed forecasts or "
                        "determine max forecast_id. Forecast_idx remains 0.",
                        extra={"log_id": ""},
                    )
                    self.completed_forecasts.clear()  # Clear any potentially incomplete state.
                    return

                initial_processed_count: int = 0
                for row in reader:
                    try:
                        model_name: str = row["model"]
                        mode_name: str = row["mode"]
                        sample_id: int = int(row["sample_id"])
                        forecast_successful_str: str = row["forecast_successful"].strip().lower()
                        is_successful: bool = forecast_successful_str == "true"

                        # Add to `completed_forecasts` based on `redo_failed_forecasts` setting.
                        if not self.redo_failed_forecasts or (self.redo_failed_forecasts and is_successful):
                            self.completed_forecasts.add((model_name, mode_name, sample_id))
                            initial_processed_count += 1
                    except ValueError:
                        logger.warning(
                            f"({self.process}) Skipping row due to invalid data in meta file (e.g., sample_id non-integer): {row}",
                            extra={"log_id": ""},
                        )
                        continue # Skip to the next row if parsing fails for this row.
                    except KeyError as ke:
                        logger.warning(
                            f"({self.process}) Skipping row due to missing expected column '{ke}' in meta file: {row}",
                            extra={"log_id": ""},
                        )
                        continue

                    # Determine max `forecast_id` from this row to update overall max.
                    try:
                        forecast_ids_str_from_row: str = row.get("forecast_ids", "[]")
                        if forecast_ids_str_from_row and forecast_ids_str_from_row.strip() != "[]":
                            # Use ast.literal_eval for safe parsing of string representation of list/tuple.
                            ids_list_parsed: Union[List[Any], Tuple[Any, ...]] = ast.literal_eval(forecast_ids_str_from_row)
                            
                            if isinstance(ids_list_parsed, (list, tuple)) and ids_list_parsed:
                                # Filter out any non-digit strings within the list before conversion and max.
                                valid_ids_in_row: List[int] = [
                                    int(id_val) for id_val in ids_list_parsed if str(id_val).isdigit()
                                ]
                                if valid_ids_in_row:
                                    current_max_id_in_row: int = max(valid_ids_in_row)
                                    max_forecast_idx_found_so_far = max(
                                        max_forecast_idx_found_so_far, current_max_id_in_row
                                    )
                    except (ValueError, SyntaxError, TypeError) as parse_err:
                        logger.warning(
                            f"({self.process}) Failed to parse 'forecast_ids' ('{forecast_ids_str_from_row}') "
                            f"from meta file row: {parse_err}. Row: {row}",
                            extra={"log_id": ""},
                        )
                    except Exception as e:
                        logger.error(
                            f"({self.process}) Unexpected error processing 'forecast_ids' from meta row: {e}. Row: {row}",
                            extra={"log_id": ""},
                        )

            # Set the main `forecast_idx` to one greater than the maximum found, for new unique IDs.
            self.forecast_idx = max_forecast_idx_found_so_far + 1 if max_forecast_idx_found_so_far > 0 else 0


            logger.info(
                f"({self.process}) Loaded {len(self.completed_forecasts)} previously recorded forecast entries "
                "to potentially skip.",
                extra={"log_id": ""},
            )
            logger.info(
                f"({self.process}) Next new forecast_id will start from {self.forecast_idx} based on meta file.",
                extra={"log_id": ""},
            )
            if self.redo_failed_forecasts:
                logger.info(
                    f"({self.process}) 'redo_failed_forecasts' is True. Unsuccessful forecasts from meta file "
                    "will be attempted again.",
                    extra={"log_id": ""},
                )
            else:
                logger.info(
                    f"({self.process}) 'redo_failed_forecasts' is False. Both successful and unsuccessful forecasts "
                    "from meta file will be skipped.",
                    extra={"log_id": ""},
                )

        except Exception as e:
            logger.exception(
                f"({self.process}) Error loading completed forecasts or max ID from {self.meta_path}: {e}. "
                "Forecast_idx remains 0. Starting fresh.",
                extra={"log_id": ""},
            )
            self.completed_forecasts.clear()
            self.forecast_idx = 0

    def forecast_single_sample(self, model_input: ModelInput, model_setup: ModelSetup) -> ModelOutput:
        """
        Performs a forecast for a single `ModelInput` using the specified `ModelSetup`.

        This is a helper method that dispatches the forecasting task to the
        appropriate underlying model client (e.g., `BaselineModels`, `TimeGPT`).

        Parameters
        ----------
        model_input : ModelInput
            The input data and parameters for the forecast.
        model_setup : ModelSetup
            The configuration details for the model to be used.

        Returns
        -------
        ModelOutput
            An object containing the forecast results and status.
        """
        # Retrieve the specific forecaster instance (e.g., BaselineModels, OtherModels).
        forecaster_instance: Union[BaselineModels, OtherModels, TimeGPT, FoundationModelAPIClient] = self.models[model_setup.model_category]

        # Prepare keyword arguments for the forecaster's `forecast` method.
        # This copies all attributes from `model_setup` except `model_category`.
        kwargs_for_forecast: Dict[str, Any] = vars(model_setup).copy()
        kwargs_for_forecast.pop('model_category') # Remove the internal category identifier.

        # Call the appropriate `forecast` method on the selected forecaster instance.
        forecast_result: Union[ModelOutput, List[ModelOutput]] = forecaster_instance.forecast(model_input, **kwargs_for_forecast)
        
        # Ensure the result is always a list of ModelOutput for consistent processing later.
        # This handles cases where BaselineModels might return a list and others return a single ModelOutput.
        if not isinstance(forecast_result, list):
            return [forecast_result] # Wrap single ModelOutput in a list.
        return forecast_result

    def store_forecasts(self) -> None:
        """
        Stores accumulated forecast results (metadata and detailed data) into CSV files.
        It clears the internal buffers (`self.forecasts` and `self.forecasts_info`)
        after successful storage.
        """
        # Initialize CSV writers, ensuring headers are written if files are new.
        data_writer = CSVWriter(self.data_path, fieldnames=DATA_FIELDNAMES)
        meta_writer = CSVWriter(self.meta_path, fieldnames=META_FIELDNAMES)

        processed_count: int = 0
        for output, info in zip(self.forecasts, self.forecasts_info):
            log_id: str = info.get("log_id", "") # Use log_id from info if available
            try:
                data_rows: List[Dict[str, Any]] = []
                forecast_ids_for_meta: List[int] = []
                forecast_time_seconds: Optional[float] = None

                if output.forecast_successful:
                    # Extract forecast time from metadata.
                    forecast_time_seconds = output.metadata.get("forecast_time_seconds", None)

                    # Handle feature importance for specific models like "TFT".
                    if info["model"] == "TFT" and output.metadata and "feature_importances" in output.metadata:
                        feature_importance_data = output.metadata["feature_importances"]
                        data_to_save: Dict[str, Any] = {
                            "model": info["model"],
                            "mode": info["mode"],
                            "sample_id": info["sample_id"],
                            "feature_importances": feature_importance_data
                        }
                        try:
                            with open(FEATURE_IMPORTANCE_PATH, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(data_to_save) + '\n')
                            logger.info(f"Stored feature importance for {info['model']}, mode {info['mode']}, sample {info['sample_id']}", extra={"log_id": log_id})
                        except Exception as e:
                            logger.error(f"Error saving feature importance for {info['model']}: {e}", extra={"log_id": log_id})

                    # Iterate over each target series within the ModelOutput.
                    for target_name, point_forecast_array in output.point_forecast.items():
                        # Retrieve the actual `ts_id` for this target series.
                        # This requires looking up Sample and TSInfo, which is a bit roundabout.
                        # A more direct way would be to pass `ts_id` into ModelOutput during its creation.
                        sample_obj = self.dl.get_sample(info["sample_id"])
                        if sample_obj is None:
                            logger.warning(f"Could not retrieve sample {info['sample_id']} to determine ts_id for target {target_name}. Skipping.", extra={"log_id": log_id})
                            continue
                        
                        ts_info_for_target = TSInfo(
                            dataset_names=[sample_obj.dataset_name],
                            ts_names=[sample_obj.ts_name] if sample_obj.ts_name != "_single_series_" else None,
                            target_names=[target_name]
                        )
                        target_ts_ids = ts_info_for_target.ts_ids
                        
                        if not target_ts_ids:
                            logger.warning(f"Could not find ts_id for target '{target_name}' in sample {info['sample_id']}. Skipping.", extra={"log_id": log_id})
                            continue
                        
                        ts_id_actual = target_ts_ids[0]

                        n_points: int = len(point_forecast_array)
                        for i in range(n_points):
                            self.forecast_idx += 1  # Increment global forecast ID for each data point.
                            try:
                                data_row: Dict[str, Any] = {
                                    "forecast_id": self.forecast_idx,
                                    "ts_id": ts_id_actual,
                                    "step": i,
                                    "point_forecast": float(point_forecast_array[i]), # Ensure float type
                                    "lower_bound": (
                                        float(output.lower_bound[target_name][i])
                                        if output.lower_bound is not None
                                        and target_name in output.lower_bound
                                        and i < len(output.lower_bound[target_name])
                                        else np.nan 
                                    ),
                                    "upper_bound": (
                                        float(output.upper_bound[target_name][i])
                                        if output.upper_bound is not None
                                        and target_name in output.upper_bound
                                        and i < len(output.upper_bound[target_name])
                                        else np.nan
                                    ),
                                }
                                data_rows.append(data_row)
                                forecast_ids_for_meta.append(self.forecast_idx)
                            except Exception as e:
                                logger.error(
                                    f"Error processing forecast data point for ts_id {ts_id_actual}, step {i}: {e}. "
                                    f"Skipping this data point.",
                                    extra={"log_id": log_id},
                                )
                                # Continue to next data point, but this `forecast_id` might be lost if it was just assigned.
                                # This indicates a potential issue if data_rows are not consistently added.
                                # For strictness, if a point fails, consider the whole sub-forecast failed.
                                # For current behavior, we continue to next point but note error.

                    data_writer.write_rows(data_rows)

                # Prepare metadata row for the current forecast.
                meta_row: Dict[str, Any] = {
                    "model": info["model"],
                    "mode": info["mode"],
                    "sample_id": info["sample_id"],
                    "forecast_successful": output.forecast_successful,
                    "forecast_time_seconds": forecast_time_seconds,
                    "error_log_id": output.error_log_id,
                    "forecast_ids": str(forecast_ids_for_meta), # Store as string representation of list.
                }
                meta_writer.write_row(meta_row)
                processed_count += 1

            except Exception as e:
                logger.exception(
                    f"Error while storing forecast data for {info['model']}, {info['mode']}, sample {info['sample_id']}: {e}. "
                    "Skipping this entry.",
                    extra={"log_id": log_id},
                )
                continue # Continue to the next forecast in the buffer if this one fails to store.

        # Clear the buffers after successful writing.
        self.forecasts.clear()
        self.forecasts_info.clear()
        logger.info(f"({self.process}) Stored {processed_count} forecasts successfully.", extra={"log_id": ""})

    def forecast_all(
        self, config: List[Tuple[List[str], List[str]]] = [(list(MODEL_SETUPS.keys()), COV_MODES)]
    ) -> None:
        """
        Executes the full forecasting process for all defined models and modes across all samples.
        It includes an enhanced skipping logic to efficiently resume interrupted runs and
        handle recurring model failures.

        Skipping Rules:
        1. **Rule 1 (Window Skip)**: Skips subsequent windows (samples with increasing `window` values)
           for a specific `(model_name, mode, dataset_name, ts_name, fc_horizon)`
           if the first window in that sequence fails.
        2. **Rule 2 (Dataset Skip)**: Skips all *remaining* samples for a given
           `(model_name, mode, dataset_name)` if the number of *consecutive* failures
           for that dataset reaches a `FAILURE_THRESHOLD_PERCENT` of its total samples,
           or `MIN_FAILURES_FOR_DATASET_SKIP`, whichever is higher.
        3. **Rule 3 (Model Setup Skip)**: Skips an entire `(model_name, mode)` configuration
           if it has caused Rule 2 to trigger for more than `MAX_DISTINCT_DATASETS_FOR_MODEL_SKIP`
           distinct datasets.
        4. **Meta File Skip**: Skips forecasts already recorded in the `meta_path` CSV,
           unless `redo_failed_forecasts` is True, in which case only successful ones are skipped.
        """
        progress_counter: int = 0  # Tracks overall progress steps.
        total_estimated_steps: int = 0 # An estimate of total steps for progress percentage.

        num_samples_total: int = len(self.samples)
        if num_samples_total == 0:
            logger.warning(f"({self.process}) No samples prepared by DataLoader. Exiting forecast_all.", extra={"log_id": ""})
            return

        # Precompute total samples per dataset for Rule 2.
        samples_per_dataset: Dict[str, int] = defaultdict(int)
        for sample in self.samples:
            samples_per_dataset[sample.dataset_name] += 1
        logger.info(f"({self.process}) Samples per dataset: {dict(samples_per_dataset)}", extra={"log_id": ""})

        # Calculate total estimated steps for progress bar.
        for model_setups_list_item, cov_modes_item in config:
            num_modes_for_config = len(cov_modes_item)
            # Each (model_setup, mode) combination will attempt to process all samples.
            total_estimated_steps += len(model_setups_list_item) * num_modes_for_config * num_samples_total

        logger.info(
            f"({self.process}) START FORECASTS: Total estimated steps to process: {total_estimated_steps}",
            extra={"log_id": ""},
        )

        # --- Reset dynamic skipping state for this new `forecast_all` run ---
        self.failed_windows.clear()
        self.consecutive_dataset_failures.clear()
        self.skipped_datasets.clear()
        self.distinct_datasets_skipped_by_rule2.clear()
        self.skipped_model_setups.clear()

        for model_setups_list, cov_modes in config:
            for model_setup_str in model_setups_list:
                if model_setup_str not in MODEL_SETUPS:
                    logger.error(f"({self.process}) Model setup '{model_setup_str}' not found in MODEL_SETUPS. Skipping this model setup.", extra={"log_id": ""})
                    # Increment progress for all samples of this model_setup across all modes in this config.
                    progress_counter += len(cov_modes) * num_samples_total
                    continue

                model_setup_obj: ModelSetup = MODEL_SETUPS[model_setup_str]

                for mode_name in cov_modes:
                    config_key: Tuple[str, str] = (model_setup_str, mode_name)

                    # --- Apply Rule 3: Skip entire model setup/mode if threshold hit ---
                    if config_key in self.skipped_model_setups:
                        logger.info(
                            f"({self.process}) Skipping model setup '{model_setup_str}' with mode '{mode_name}' - "
                            f"threshold of {MAX_DISTINCT_DATASETS_FOR_MODEL_SKIP} failed distinct datasets reached.",
                            extra={"log_id": ""},
                        )
                        # Increment progress for all remaining samples for this specific (model, mode) combo.
                        progress_counter += num_samples_total
                        continue

                    # Reset dataset-level counters for this configuration, as it's a new (model, mode) combination.
                    # This is implicitly handled by `defaultdict` initialization and resetting on success below.

                    self.dl.change_mode(mode_name) # Set the DataLoader's covariate mode.

                    # Apply max_ts_length settings based on model setup.
                    # This logic should ideally be managed within ModelSetup or ModelInput.
                    if model_setup_str in ["all_baseline", "arima", "ttm_finetuned", "TFT", "timegpt_finetuned"]:
                        self.dl.set_max_ts_length(10000)
                    else:
                        self.dl.set_max_ts_length(None)

                    # Iterate through samples provided by the DataLoader for the current mode.
                    for sample_index, model_input_obj in enumerate(self.dl):
                        sample_obj = self.dl.samples[sample_index] # Get the original Sample object
                        
                        current_dataset_name: str = sample_obj.dataset_name
                        current_ts_name: str = sample_obj.ts_name
                        current_forecast_horizon: int = sample_obj.fc_horizon

                        # Define keys for tracking and skipping checks.
                        # `window_key` uniquely identifies a specific time series forecast window.
                        window_key: Tuple[str, str, str, str, int] = (
                            model_setup_str,
                            mode_name,
                            current_dataset_name,
                            current_ts_name,
                            current_forecast_horizon,
                        )
                        # `dataset_key` uniquely identifies a model-mode-dataset combination.
                        dataset_key: Tuple[str, str, str] = (
                            model_setup_str,
                            mode_name,
                            current_dataset_name,
                        )

                        # ============================================
                        # --- Apply Skipping Checks (Order Matters) ---
                        # ============================================

                        # 1. Check Rule 3 (Model Setup Skip) - Outer loop already handles, this is redundant here.

                        # 2. Check Rule 2 (Dataset Skip) - Skip remaining samples for a dataset.
                        if dataset_key in self.skipped_datasets:
                            # Log less frequently to avoid flood if many samples are skipped.
                            if progress_counter % STORE_FREQUENCY * 5 == 0:
                                logger.info(
                                    f"({self.process}) Skipping dataset '{current_dataset_name}' for model '{model_setup_str}', mode '{mode_name}' - "
                                    "Dataset skip threshold reached.",
                                    extra={"log_id": ""},
                                )
                            progress_counter += 1
                            continue # Skip to the next sample.

                        # 3. Check Rule 1 (Window Skip) - Skip subsequent windows after first failure.
                        if window_key in self.failed_windows:
                            if progress_counter % STORE_FREQUENCY == 0:
                                logger.info(
                                    f"({self.process}) Skipping forecast window {window_key} - "
                                    "Previous window for this combination failed.",
                                    extra={"log_id": ""},
                                )
                            progress_counter += 1
                            continue # Skip to the next sample.

                        # 4. Check Meta File (Previously Completed/Skipped).
                        # For baseline models, a single `model_setup_str` corresponds to multiple actual models.
                        # We need to check if ALL of the constituent baseline models (e.g., AutoARIMA, AutoETS)
                        # have been completed for this specific (mode, sample_id) before skipping.
                        specific_models_to_check: List[str] = []
                        if model_setup_obj.model_category == "baseline":
                            # Retrieve the list of specific models (e.g., ["AutoARIMA", "AutoETS"])
                            # from the `model_setup_obj`.
                            specific_models_to_check.extend(getattr(model_setup_obj, 'models', [model_setup_str]))
                        else:
                            specific_models_to_check.append(model_setup_str)

                        should_skip_from_meta: bool = True
                        for specific_model_name in specific_models_to_check:
                            if (specific_model_name, mode_name, sample_obj.id) not in self.completed_forecasts:
                                should_skip_from_meta = False
                                break # If any constituent model is not completed, do not skip.
                        
                        if should_skip_from_meta:
                            progress_counter += 1
                            continue # Skip to the next sample if already completed.

                        # ========================
                        # --- Perform Forecast ---
                        # ========================
                        log_id_forecast: str = str(uuid.uuid4()) # Unique ID for this forecast attempt.
                        logger.info(
                            f"({self.process}) FORECAST: Model: {model_setup_str}, Mode: {mode_name}, Sample ID: {sample_obj.id}",
                            extra={"log_id": log_id_forecast},
                        )
                        
                        forecast_succeeded_overall: bool = False # Flag to track overall success for this step.
                        try:
                            # Store current forecast/info list lengths to identify newly added items.
                            forecast_count_before_step: int = len(self.forecasts)
                            info_count_before_step: int = len(self.forecasts_info)

                            # `forecast_single_sample` now returns a List[ModelOutput].
                            list_of_model_outputs: List[ModelOutput] = self.forecast_single_sample(model_input_obj, model_setup_obj)

                            # Determine overall success for this sample/model/mode combination.
                            step_had_outputs: bool = bool(list_of_model_outputs)
                            if step_had_outputs:
                                # Overall success if ALL returned outputs are successful.
                                forecast_succeeded_overall = all(output.forecast_successful for output in list_of_model_outputs)
                            else:
                                forecast_succeeded_overall = False # No outputs returned implies failure.

                            # --- Append results and info to buffers ---
                            # Append ModelOutput instances to the `self.forecasts` buffer.
                            self.forecasts.extend(list_of_model_outputs)

                            # For each ModelOutput, append a corresponding info dictionary.
                            # Ensure `info` reflects the specific model name if it's a baseline.
                            for output_item in list_of_model_outputs:
                                specific_model_name_for_info: str = model_setup_str
                                if model_setup_obj.model_category == "baseline" and hasattr(output_item.metadata, 'model_name_from_output'):
                                    # If baseline, try to get specific model name from output metadata if available.
                                    specific_model_name_for_info = output_item.metadata.get('model_name_from_output', model_setup_str)
                                
                                self.forecasts_info.append(
                                    {
                                        "model": specific_model_name_for_info,
                                        "mode": mode_name,
                                        "sample_id": sample_obj.id,
                                        "log_id": output_item.error_log_id if not output_item.forecast_successful else log_id_forecast # Use specific error ID if failed.
                                    }
                                )

                        except Exception as e:
                            logger.error(
                                f"({self.process}) Uncaught error during forecast for Model: {model_setup_str}, Mode: {mode_name}, Sample ID: {sample_obj.id}: {e}",
                                exc_info=True,
                                extra={"log_id": log_id_forecast},
                            )
                            forecast_succeeded_overall = False
                            # If an uncaught exception occurred and no info was added, add a placeholder.
                            if len(self.forecasts_info) == info_count_before_step:
                                self.forecasts_info.append(
                                    {
                                        "model": model_setup_str,
                                        "mode": mode_name,
                                        "sample_id": sample_obj.id,
                                        "log_id": log_id_forecast,
                                    }
                                )

                        # ===============================================
                        # --- Update Failure Tracking & Apply Rules ---
                        # ===============================================
                        if not forecast_succeeded_overall:
                            # Rule 1 Trigger: Mark this window combination as failed.
                            self.failed_windows.add(window_key)

                            # Rule 2 Update: Increment consecutive failures for the dataset.
                            self.consecutive_dataset_failures[dataset_key] += 1

                            # Rule 2 Check: Skip dataset if threshold met.
                            dataset_total_samples = samples_per_dataset.get(current_dataset_name, 0)
                            # Minimum failures needed to trigger a skip for this dataset.
                            min_failures_to_trigger_skip: int = max(
                                MIN_FAILURES_FOR_DATASET_SKIP,
                                int(FAILURE_THRESHOLD_PERCENT * dataset_total_samples),
                            )

                            if (
                                self.consecutive_dataset_failures[dataset_key] >= min_failures_to_trigger_skip
                                and dataset_key not in self.skipped_datasets
                            ):
                                logger.warning(
                                    f"({self.process}) Dataset skip threshold ({min_failures_to_trigger_skip} failures "
                                    f"out of {dataset_total_samples} samples) reached for dataset '{current_dataset_name}' "
                                    f"with Model: '{model_setup_str}', Mode: '{mode_name}'. Skipping remaining samples for this dataset.",
                                    extra={"log_id": log_id_forecast},
                                )
                                self.skipped_datasets.add(dataset_key)

                                # Rule 3 Update: Add dataset to skipped set for this config.
                                self.distinct_datasets_skipped_by_rule2[config_key].add(current_dataset_name)

                                # Rule 3 Check: Skip model setup if enough distinct datasets failed.
                                if (
                                    len(self.distinct_datasets_skipped_by_rule2[config_key])
                                    > MAX_DISTINCT_DATASETS_FOR_MODEL_SKIP
                                ):
                                    logger.warning(
                                        f"({self.process}) Model setup config '{model_setup_str}' with mode '{mode_name}' "
                                        f"skipped - more than {MAX_DISTINCT_DATASETS_FOR_MODEL_SKIP} distinct datasets "
                                        f"failed ({self.distinct_datasets_skipped_by_rule2[config_key]}).",
                                        extra={"log_id": log_id_forecast},
                                    )
                                    self.skipped_model_setups.add(config_key)
                        else:  # Forecast Succeeded.
                            # Reset consecutive failure count for this dataset, as a success broke the streak.
                            self.consecutive_dataset_failures[dataset_key] = 0

                        # ========================
                        # --- Progress & Store ---
                        # ========================
                        progress_counter += 1
                        # Store forecasts periodically to prevent data loss.
                        if progress_counter % STORE_FREQUENCY == 0 and len(self.forecasts) > 0:
                            progress_percent = (
                                progress_counter / total_estimated_steps * 100 if total_estimated_steps > 0 else 0
                            )
                            logger.info(
                                f"({self.process}) PROGRESS: {progress_percent:.2f}% ({progress_counter}/{total_estimated_steps} steps processed)",
                                extra={"log_id": ""},
                            )
                            self.store_forecasts()

                    # --- End of Sample Loop ---
                    # Store any remaining forecasts for this specific (model_setup, mode)
                    # combination before moving to the next one.
                    if len(self.forecasts) > 0:
                        logger.info(
                            f"({self.process}) Storing remaining forecasts for Model: '{model_setup_str}', Mode: '{mode_name}'...",
                            extra={"log_id": ""},
                        )
                        self.store_forecasts()

            # --- End of Model Setup Loop ---
        # --- End of Config Loop ---

        # --- Final Store ---
        # Ensure any last remaining forecasts are stored at the very end of the entire process.
        if len(self.forecasts) > 0:
            logger.info(f"({self.process}) Storing final remaining forecasts...", extra={"log_id": ""})
            self.store_forecasts()

        logger.info(
            f"({self.process}) PROGRESS: 100.00% ({progress_counter} actual steps processed)",
            extra={"log_id": ""},
        )
        logger.info(f"({self.process}) FINISHED FORECASTS", extra={"log_id": ""})
        logger.info(
            f"({self.process}) Final skip counts: Windows={len(self.failed_windows)}, Datasets={len(self.skipped_datasets)}, ModelConfigs={len(self.skipped_model_setups)}",
            extra={"log_id": ""},
        )
        if self.skipped_model_setups:
            logger.info(
                f"({self.process}) Skipped Model Configs (Model, Mode): {self.skipped_model_setups}",
                extra={"log_id": ""},
            )