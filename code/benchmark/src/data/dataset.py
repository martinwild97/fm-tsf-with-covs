import ast
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd


# --- Path Configuration ---
DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data"
DATASET_INFO_CSV: Path = DATA_DIR / "dataset_info.csv"
DATASET_DIR: Path = DATA_DIR / "prepared_datasets"
TIMESERIES_INFO_CSV: Path = DATA_DIR / "timeseries_info.csv"


# --- Logger Setup ---
logger: logging.Logger = logging.getLogger("Dataset")
ts_info_logger: logging.Logger = logging.getLogger("TimeseriesInfo")


class Dataset:
    """
    Represents a full dataset, loading metadata from `dataset_info.csv`
    and the time series data itself from a corresponding CSV file.

    Attributes
    ----------
    dataset_name : str
        The unique name of the dataset.
    domain : Optional[str]
        The domain or category of the dataset (e.g., 'energy', 'finance').
    granularity : Optional[str]
        The time granularity of the series (e.g., 'H' for hourly, 'D' for daily).
    ts_length : int
        The total length of the longest time series in the dataset.
    num_ts : int
        The number of individual time series within the dataset.
    ts_names : List[str]
        A list of names for individual time series if the dataset contains multiple.
    targets : List[str]
        A list of column names representing the target variables for forecasting.
    num_targets : int
        The number of target variables.
    past_covariates : List[str]
        A list of column names for covariates known in the past and present.
    num_past_covariates : int
        The number of past covariates.
    future_covariates : List[str]
        A list of column names for covariates known in the past, present, and future.
    num_future_covariates : int
        The number of future covariates.
    categorical_covariates : List[str]
        A list of column names for categorical covariates.
    num_categorical_covariates : int
        The number of categorical covariates.
    df : pd.DataFrame
        The loaded pandas DataFrame containing the time series data.

    Raises
    ------
    FileNotFoundError
        If `dataset_info.csv` or the specific dataset's CSV file is not found.
    ValueError
        If no metadata is found for the specified `dataset_name`.
    """

    # --- Metadata Attributes (type hints for clarity) ---
    dataset_name: str
    domain: Optional[str]
    granularity: Optional[str]
    ts_length: int
    num_ts: int
    ts_names: List[str]
    targets: List[str]
    num_targets: int
    past_covariates: List[str]
    num_past_covariates: int
    future_covariates: List[str]
    num_future_covariates: int
    categorical_covariates: List[str]
    num_categorical_covariates: int

    # --- Data Attribute ---
    df: pd.DataFrame

    def __init__(self, dataset_name: str) -> None:
        """
        Initializes the Dataset object by loading metadata and the actual time series data.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset. This must match an entry in `dataset_info.csv`
            and the filename (e.g., 'my_dataset.csv').
        """
        self.dataset_name = dataset_name
        logger.debug(f"Initializing Dataset object for '{self.dataset_name}'")

        # Initialize all expected metadata attributes to default empty/None values
        # This prevents AttributeError if metadata columns are missing.
        for attr_name, attr_type in self.__annotations__.items():
            if attr_name not in ["dataset_name", "df"]:  # Exclude self.dataset_name and self.df
                if "List" in str(attr_type):
                    setattr(self, attr_name, [])
                elif "Optional" in str(attr_type) and "str" in str(attr_type):
                    setattr(self, attr_name, None)
                elif "int" in str(attr_type):
                    setattr(self, attr_name, 0)
                logger.debug(f"Initialized default for {attr_name}")

        self._load_metadata()
        self._load_data()

        logger.debug(f"Successfully loaded dataset '{self.dataset_name}'")

    def _load_metadata(self) -> None:
        """
        Loads metadata for the dataset from the `dataset_info.csv` file.

        This method populates the Dataset object's attributes based on the
        corresponding entry in the metadata CSV. It handles parsing of
        list-like strings and type conversions.

        Raises
        ------
        FileNotFoundError
            If `dataset_info.csv` is not found.
        ValueError
            If no metadata entry is found for the specified `dataset_name`.
        """
        logger.debug(f"Attempting to load metadata from: {DATASET_INFO_CSV}")
        try:
            dataset_info_full_df: pd.DataFrame = pd.read_csv(DATASET_INFO_CSV)
        except FileNotFoundError:
            logger.critical(
                f"Metadata file not found at path: {DATASET_INFO_CSV}", exc_info=True
            )
            raise FileNotFoundError(
                f"Error: Metadata CSV not found at path '{DATASET_INFO_CSV}'"
            )
        except Exception as e:
            logger.critical(f"Error reading metadata CSV: {e}", exc_info=True)
            raise

        dataset_rows: pd.DataFrame = dataset_info_full_df[
            dataset_info_full_df["dataset_name"] == self.dataset_name
        ]

        if dataset_rows.empty:
            logger.error(
                f"No metadata found for dataset '{self.dataset_name}' in {DATASET_INFO_CSV}."
            )
            raise ValueError(f"No metadata found for dataset '{self.dataset_name}'.")
        if len(dataset_rows) > 1:
            logger.warning(
                f"Multiple metadata entries found for dataset '{self.dataset_name}'. "
                "Using the first one."
            )

        dataset_info: dict = dataset_rows.iloc[0].to_dict()
        logger.debug(f"Raw metadata found: {dataset_info}")

        list_columns: List[str] = [
            "ts_names",
            "targets",
            "past_covariates",
            "future_covariates",
            "categorical_covariates",
        ]

        # Use __annotations__ to get expected attribute types
        expected_attributes: Dict[str, type] = self.__class__.__annotations__

        for key, value in dataset_info.items():
            if key in expected_attributes:
                processed_value: Any = None
                target_type = expected_attributes.get(key)

                if key in list_columns:
                    if isinstance(value, str):
                        try:
                            processed_value = ast.literal_eval(value)
                            if not isinstance(processed_value, list):
                                logger.warning(
                                    f"Column '{key}' for dataset '{self.dataset_name}' "
                                    f"did not evaluate to a list. Keeping original value: {value}"
                                )
                                processed_value = value
                        except (ValueError, SyntaxError) as e:
                            logger.warning(
                                f"Error evaluating list string for column '{key}' in dataset "
                                f"'{self.dataset_name}': '{value}'. Defaulting to empty list. Error: {e}"
                            )
                            processed_value = []
                    elif pd.isna(value):
                        processed_value = []
                    else:
                        processed_value = value # Keep original if not string and not NaN
                else:
                    if pd.isna(value):
                        # Handle NaN for non-list columns based on expected type
                        if target_type == int:
                            processed_value = 0
                        elif target_type == str:
                            processed_value = "" # Or None, depending on desired empty string vs None
                        else:
                            processed_value = None
                    else:
                        # Attempt type conversion for non-list columns
                        current_value_type = type(value)
                        needs_conversion = False
                        expected_py_type: Optional[type] = None

                        if target_type == int and not isinstance(value, (int, np.integer)):
                            needs_conversion = True
                            expected_py_type = int
                        elif target_type == str and not isinstance(value, str):
                            needs_conversion = True
                            expected_py_type = str
                        elif target_type == float and not isinstance(value, (float, np.floating)):
                            needs_conversion = True
                            expected_py_type = float

                        if needs_conversion and expected_py_type is not None:
                            try:
                                processed_value = expected_py_type(value)
                            except (ValueError, TypeError):
                                logger.warning(
                                    f"Could not convert value '{value}' (type: {current_value_type}) "
                                    f"to type {target_type} for key '{key}'. Keeping original."
                                )
                                processed_value = value
                        else:
                            processed_value = value

                setattr(self, key, processed_value)

        logger.debug(f"Finished setting metadata attributes for '{self.dataset_name}'.")

    def _load_data(self) -> None:
        """
        Loads the actual time series data for the dataset from its CSV file.

        This method populates the `df` attribute of the Dataset object,
        converts 'ts_name' to string, and attempts to convert all non-date/ts_name
        columns to numeric, coercing errors to NaN.

        Raises
        ------
        FileNotFoundError
            If the dataset's CSV file is not found.
        """
        file_path: Path = DATASET_DIR / f"{self.dataset_name}.csv"
        logger.debug(f"Attempting to load dataset file from: {file_path}")

        try:
            self.df = pd.read_csv(file_path, sep=";")
            logger.debug(
                f"Loaded DataFrame with shape {self.df.shape} for '{self.dataset_name}'"
            )
        except FileNotFoundError:
            logger.error(f"Dataset file not found at path: {file_path}", exc_info=True)
            raise FileNotFoundError(f"Error: Dataset CSV not found at path '{file_path}'")
        except Exception as e:
            logger.critical(
                f"Error reading dataset CSV '{file_path}': {e}", exc_info=True
            )
            raise

        if "ts_name" in self.df.columns:
            logger.debug(f"Converting 'ts_name' column to string for '{self.dataset_name}'.")
            self.df["ts_name"] = self.df["ts_name"].astype(str)

        logger.debug(
            f"Attempting to convert non-date/ts_name columns to numeric for '{self.dataset_name}'."
        )
        for col in self.df.columns:
            if col not in ["ts_name", "date"]:
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    original_nan_count: int = self.df[col].isnull().sum()
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        new_nan_count: int = self.df[col].isnull().sum()
                        if new_nan_count > original_nan_count:
                            logger.debug(
                                f"Converted column '{col}' to numeric for '{self.dataset_name}'. "
                                f"Introduced {new_nan_count - original_nan_count} new NaN values."
                            )

    def __str__(self) -> str:
        """
        Provides a concise string representation of the Dataset object.
        """
        domain: str = getattr(self, "domain", "N/A")
        granularity: str = getattr(self, "granularity", "N/A")
        shape: str = str(self.df.shape) if hasattr(self, "df") else "N/A"
        num_ts: int = getattr(self, "num_ts", 0)
        num_targets: int = getattr(self, "num_targets", 0)
        return (
            f"Dataset(name='{self.dataset_name}', domain='{domain}', granularity='{granularity}', "
            f"shape={shape}, num_ts={num_ts}, num_targets={num_targets})"
        )

    def __repr__(self) -> str:
        """
        Provides a more detailed string representation suitable for debugging.
        """
        return self.__str__()