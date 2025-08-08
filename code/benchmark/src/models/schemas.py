from typing import List, Optional, Dict, Tuple, Union, Any
from pydantic import BaseModel, ConfigDict
import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame


# --- Constants for ModelInput ---
DEFAULT_DATE_COLUMN: str = "date"
DEFAULT_QUANTILE_LEVEL: float = 0.9
NIXTLA_TIMESTAMP_COLUMN: str = "ds"
NIXTLA_UNIQUE_ID_COLUMN: str = "unique_id"
NIXTLA_TARGET_VALUE_COLUMN: str = "y"
AUTOGLUON_TARGET_VALUE_COLUMN: str = "y"
AUTOGLUON_ITEM_ID_COLUMN: str = "item_id"
API_FORECAST_HORIZON_KEY: str = "forecast_horizon"
API_ACTUALS_KEY: str = "actuals"
API_TIMESTAMP_KEY: str = "timestamp"
API_DATA_KEY: str = "data"
API_TIMESTAMP_FORMAT_KEY: str = "timestamp_format"
API_PAST_COV_NAMES_KEY: str = "past_cov_names"
API_FUTURE_COV_NAMES_KEY: str = "future_cov_names"
API_CATEGORICAL_COV_NAMES_KEY: str = "categorical_cov_names"


# --- Constants for ModelOutput ---
MODEL_OUTPUT_POINT_FORECAST_KEY: str = "point_forecast"
MODEL_OUTPUT_LOWER_BOUND_KEY: str = "lower_bound"
MODEL_OUTPUT_UPPER_BOUND_KEY: str = "upper_bound"
MODEL_OUTPUT_FORECAST_SUCCESSFUL_KEY: str = "forecast_successful"
MODEL_OUTPUT_ERROR_LOG_ID_KEY: str = "error_log_id"
MODEL_OUTPUT_METADATA_KEY: str = "metadata"


class ModelInput(BaseModel):
    """
    Schema for input data to forecasting models.

    This class defines the structure for time series data and related parameters
    that are passed to different forecasting model wrappers. It includes methods
    to convert the input DataFrame into formats expected by specific libraries
    like Nixtla, AutoGluon, and the FM API.
    """

    df: pd.DataFrame
    forecast_horizon: int
    targets: Optional[List[str]] = []
    past_covs: Optional[List[str]] = []
    future_covs: Optional[List[str]] = []
    categorical_covs: Optional[List[str]] = []
    date_col: Optional[str] = DEFAULT_DATE_COLUMN
    frequency: Optional[str] = None
    timestamp_format: Optional[str] = None
    quantile: Optional[float] = DEFAULT_QUANTILE_LEVEL

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_nixtla_input(self, include_past_covs: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Converts the input data to the long format expected by Nixtla forecasting packages.

        The main DataFrame (`df`) is truncated to exclude the forecast horizon period,
        and targets are melted into a single 'y' column with 'unique_id' for series identification.
        Future covariates (`X_df`) are extracted for the forecast horizon period.

        Parameters
        ----------
        include_past_covs : bool, default=False
            If True, past covariates are kept in the main DataFrame (`df`) after melting.
            If False, past covariates are dropped from `df` before melting.
            Note: For `NixtlaClient.forecast` `hist_exog_list` parameter, it expects `X_df` to
            contain *all* exogenous variables, including past covariates, up to `h` values.
            This method's `X_df` only includes future covariates by default.
            The `df` includes `past_covs` if `include_past_covs` is True.

        Returns
        -------
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]
            A tuple containing:
            - `df`: The training DataFrame in Nixtla long format (`ds`, `unique_id`, `y`, plus covariates).
            - `X_df`: The future covariates DataFrame in Nixtla long format (`ds`, `unique_id`, plus future_covs),
                      or None if no future covariates are present.
        """
        df_copy: pd.DataFrame = self.df.copy()

        # Convert date column to datetime and rename for Nixtla format
        df_copy[self.date_col] = pd.to_datetime(df_copy[self.date_col], format=self.timestamp_format)
        df_copy = df_copy.rename(columns={self.date_col: NIXTLA_TIMESTAMP_COLUMN})

        x_df_nixtla: Optional[pd.DataFrame] = None
        if self.future_covs:
            # Extract future covariates for the forecast horizon
            x_df_nixtla = df_copy.tail(self.forecast_horizon).copy()
            x_df_nixtla = x_df_nixtla[[NIXTLA_TIMESTAMP_COLUMN] + self.future_covs]
            
            # Replicate future covariates for each target series to match Nixtla's long format for X_df
            x_df_nixtla = x_df_nixtla.loc[x_df_nixtla.index.repeat(len(self.targets))]
            x_df_nixtla[NIXTLA_UNIQUE_ID_COLUMN] = self.targets * self.forecast_horizon
            x_df_nixtla = x_df_nixtla.reset_index(drop=True)

        # Truncate training data to exclude the forecast horizon
        df_training: pd.DataFrame = df_copy.head(len(df_copy) - self.forecast_horizon).copy()

        # Define columns to melt for the training data based on covariate inclusion
        id_vars_melt: List[str] = [NIXTLA_TIMESTAMP_COLUMN]
        if include_past_covs:
            id_vars_melt.extend(self.past_covs)
        id_vars_melt.extend(self.future_covs) # Future covariates are also in training period

        # Perform melt operation to transform target columns into a single 'y' column
        df_melted: pd.DataFrame = df_training.melt(
            id_vars=id_vars_melt,
            value_vars=self.targets,
            var_name=NIXTLA_UNIQUE_ID_COLUMN,
            value_name=NIXTLA_TARGET_VALUE_COLUMN,
        )
        return df_melted, x_df_nixtla

    def get_autogluon_input(self) -> TimeSeriesDataFrame:
        """
        Converts the input data to the `TimeSeriesDataFrame` format expected by AutoGluon.

        This involves:
        1. Copying the original DataFrame.
        2. Converting the date column to datetime objects.
        3. Setting future target values to NaN for the forecast horizon period,
           as AutoGluon expects missing target values for the prediction phase.
        4. Melting target columns into a single 'y' column, with 'item_id'
           to distinguish individual time series.
        5. Creating a `TimeSeriesDataFrame` with the correct timestamp column.

        Returns
        -------
        TimeSeriesDataFrame
            An AutoGluon `TimeSeriesDataFrame` prepared for forecasting.
        """
        df_copy: pd.DataFrame = self.df.copy()

        # Convert date column to datetime.
        df_copy[self.date_col] = pd.to_datetime(df_copy[self.date_col], format=self.timestamp_format)

        # Set target values to None (NaN) for the forecast horizon period.
        # AutoGluon expects missing target values in the forecast horizon for prediction.
        for cov in self.targets + self.past_covs:
            if cov in df_copy.columns: # Ensure column exists before trying to modify
                df_copy.loc[df_copy.index[-self.forecast_horizon:], cov] = np.nan

        # Melt the DataFrame to long format: 'item_id' for series, 'y' for values.
        # Include all relevant covariates in `id_vars`.
        id_vars_melt: List[str] = [self.date_col] + self.past_covs + self.future_covs + self.categorical_covs
        
        # Filter id_vars_melt to ensure only columns present in df_copy are used.
        # This prevents KeyError if a covariate list contains a name not in df_copy.
        id_vars_melt_filtered = [col for col in id_vars_melt if col in df_copy.columns]

        df_melted: pd.DataFrame = df_copy.melt(
            id_vars=id_vars_melt_filtered,
            value_vars=self.targets,
            var_name=AUTOGLUON_ITEM_ID_COLUMN,
            value_name=AUTOGLUON_TARGET_VALUE_COLUMN,
        )
        # Create and return TimeSeriesDataFrame.
        autogluon_df: TimeSeriesDataFrame = TimeSeriesDataFrame(df_melted, timestamp_column=self.date_col)
        return autogluon_df

    def get_fm_api_input(self) -> Dict[str, Any]:
        """
        Converts the input data to the JSON-serializable format expected by the FM API.

        The API expects a dictionary with `forecast_horizon` and `actuals`.
        `actuals` contains `timestamp`, `data` (a dictionary of series values keyed by column name),
        and optional `past_cov_names`, `future_cov_names`, `categorical_cov_names`.

        Returns
        -------
        Dict[str, Any]
            A dictionary structured for the FM API request.
        """
        df_copy: pd.DataFrame = self.df.copy()

        # Extract timestamps. If no future covariates are included, timestamps are truncated.
        timestamps: List[Any]
        if not self.future_covs:
            timestamps = df_copy[self.date_col].tolist()[:-self.forecast_horizon]
        else:
            timestamps = df_copy[self.date_col].tolist()

        # Prepare the 'data' dictionary: all columns except timestamp, converted to lists.
        values: Dict[str, List[Union[float, int, str]]] = df_copy.to_dict(orient="list")
        values.pop(self.date_col, None)  # Remove date column from data, safely using .pop(key, None)

        # Truncate values for past covariates and targets if no future covariates are present.
        # This ensures that 'actuals' only contains historical data up to the point of forecast start.
        if not self.future_covs:
            for key in values:
                if key in self.past_covs or key in self.targets:
                    values[key] = values[key][:-self.forecast_horizon]

        request_data: Dict[str, Any] = {
            API_FORECAST_HORIZON_KEY: self.forecast_horizon,
            API_ACTUALS_KEY: {
                API_TIMESTAMP_KEY: timestamps,
                API_DATA_KEY: values,
                API_TIMESTAMP_FORMAT_KEY: self.timestamp_format,
            },
        }

        # Add covariate names lists if they are populated.
        if self.past_covs:
            request_data[API_ACTUALS_KEY][API_PAST_COV_NAMES_KEY] = self.past_covs
        if self.future_covs:
            request_data[API_ACTUALS_KEY][API_FUTURE_COV_NAMES_KEY] = self.future_covs
        if self.categorical_covs:
            request_data[API_ACTUALS_KEY][API_CATEGORICAL_COV_NAMES_KEY] = self.categorical_covs

        return request_data


class ModelOutput(BaseModel):
    """
    Schema for output data from forecasting models.

    This class encapsulates the results of a forecast, including point predictions,
    prediction intervals, and metadata about the forecast process (e.g., success status, time taken).
    """

    point_forecast: Optional[Dict[str, np.ndarray]] = None
    lower_bound: Optional[Dict[str, np.ndarray]] = None
    upper_bound: Optional[Dict[str, np.ndarray]] = None
    forecast_successful: bool
    error_log_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def from_df(
        self,
        df: pd.DataFrame,
        quantile_keys: Tuple[str, str, str],
        target_key: str = NIXTLA_UNIQUE_ID_COLUMN,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ModelOutput":
        """
        Converts a DataFrame containing forecast results into a `ModelOutput` instance.

        This method is designed to parse typical forecast DataFrames (e.g., from Nixtla, AutoGluon)
        into the structured `point_forecast`, `lower_bound`, and `upper_bound` dictionaries.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the forecast results. It should have columns for
            individual time series identifiers, point forecasts, and optionally
            lower and upper bounds of prediction intervals.
        quantile_keys : Tuple[str, str, str]
            A tuple of three strings representing the column names in `df` for:
            (point forecast column, lower bound column, upper bound column).
            Example: `("TimeGPT", "TimeGPT-lo-90", "TimeGPT-hi-90")` or `("0.5", "0.05", "0.95")`.
        target_key : str, default=NIXTLA_UNIQUE_ID_COLUMN
            The name of the column in `df` that identifies individual time series (e.g., 'unique_id', 'item_id').
        metadata : Optional[Dict[str, Any]], default=None
            Optional dictionary of additional metadata to store with the `ModelOutput` (e.g., forecast time).

        Returns
        -------
        ModelOutput
            The `ModelOutput` instance populated with data from the DataFrame.
        """
        self.point_forecast = {}
        self.lower_bound = {}
        self.upper_bound = {}

        
        has_lower_bound_col: bool = quantile_keys[1] in df.columns
        has_upper_bound_col: bool = quantile_keys[2] in df.columns

        # Iterate over each unique time series ID (target) in the DataFrame.
        for target_id in df[target_key].unique():
            # Filter the DataFrame for the current time series.
            series_df: pd.DataFrame = df[df[target_key] == target_id]

            # Extract point forecast values.
            self.point_forecast[target_id] = series_df[quantile_keys[0]].values.astype(np.float32)

            # Extract lower and upper bound values if columns exist.
            if has_lower_bound_col:
                self.lower_bound[target_id] = series_df[quantile_keys[1]].values.astype(np.float32)
            if has_upper_bound_col:
                self.upper_bound[target_id] = series_df[quantile_keys[2]].values.astype(np.float32)

        self.forecast_successful = True
        self.metadata = metadata if metadata is not None else {} # Ensure metadata is a dict, not None
        return self