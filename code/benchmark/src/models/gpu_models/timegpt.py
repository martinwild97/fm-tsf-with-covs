import logging
import os
import time
import uuid
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd # Used in example, implicitly by NixtlaClient for DataFrames
from nixtla import NixtlaClient

from src.models.schemas import ModelInput, ModelOutput

# --- Constants ---
# Environment variable for Nixtla API key
NIXTLA_API_KEY_ENV_VAR: str = 'NIXTLA_API_KEY'

# Logger name
LOGGER_NAME: str = "Forecast"

# TimeGPT model names and parameters
TIMEGPT_SHORT_HORIZON_MODEL: str = "timegpt-1"
TIMEGPT_LONG_HORIZON_MODEL: str = "timegpt-1-long-horizon"
LONG_HORIZON_THRESHOLD: int = 60 # Threshold for switching to long-horizon model
FINETUNE_STEPS_DEFAULT: int = 20
FINETUNE_DEPTH_DEFAULT: int = 4

# Metadata key for forecast time
FORECAST_TIME_METADATA_KEY: str = "forecast_time_seconds"

# Column names for ModelOutput from Nixtla forecasts
TIMEGPT_POINT_FORECAST_COL_PREFIX: str = "TimeGPT"
TIMEGPT_LOWER_BOUND_COL_PREFIX: str = "TimeGPT-lo"
TIMEGPT_UPPER_BOUND_COL_PREFIX: str = "TimeGPT-hi"


# Initialize a common logger for all forecasting models.
logger: logging.Logger = logging.getLogger(LOGGER_NAME)


class TimeGPT:
    """
    TimeGPT forecasting model wrapper for Nixtla's API.
    Handles time series forecasting with optional exogenous variables and finetuning.
    """

    def __init__(self) -> None:
        """
        Initializes the TimeGPT class by setting up the Nixtla client.
        The API key is retrieved from the environment variable specified by `NIXTLA_API_KEY_ENV_VAR`.
        """
        api_key: Optional[str] = os.getenv(NIXTLA_API_KEY_ENV_VAR)
        if not api_key:
            logger.critical(
                f"{NIXTLA_API_KEY_ENV_VAR} environment variable not set. Please set your API key.",
                extra={"log_id": ""},  # No specific log_id for initialization error
            )
            self.client: Optional[NixtlaClient] = None
            return # Exit early if no API key

        try:
            self.client: NixtlaClient = NixtlaClient(api_key=api_key)
            logger.info("Nixtla client initialized successfully", extra={"log_id": ""})
        except Exception as e:
            logger.critical(
                f"Failed to initialize Nixtla client: {e}",
                extra={"log_id": ""},
                exc_info=True,  # Log full traceback for initialization errors
            )
            self.client = None

    def forecast(self, model_input: ModelInput, finetune: bool = False) -> ModelOutput:
        """
        Generates forecasts using the Nixtla TimeGPT API.

        This method supports using past and future covariates (if provided in `model_input`)
        and can optionally apply finetuning to the model. It automatically selects
        between 'timegpt-1' and 'timegpt-1-long-horizon' based on the `forecast_horizon`.

        Parameters
        ----------
        model_input : ModelInput
            An instance of `ModelInput` containing the time series data,
            forecast horizon, and other relevant metadata like covariates and quantile level.
        finetune : bool, default=False
            If True, applies finetuning steps to the TimeGPT model before forecasting.

        Returns
        -------
        ModelOutput
            A `ModelOutput` instance containing the forecast results (point forecast,
            lower bound, upper bound), success status, and forecast time.
            If the forecast fails, `forecast_successful` will be False and `error_log_id`
            will contain a UUID for logging.
        """
        # Check if the Nixtla client was successfully initialized.
        if self.client is None:
            log_id: str = str(uuid.uuid4())
            logger.error(
                "Cannot generate forecast: Nixtla client not initialized due to previous error.",
                extra={"log_id": log_id},
            )
            return ModelOutput(forecast_successful=False, error_log_id=log_id)

        try:
            # Nixtla API expects confidence level as an integer percentage (e.g., [90]).
            prediction_levels: List[int] = [int(model_input.quantile * 100)]

            # Prepare the input DataFrames for NixtlaClient.
            # `include_past_covs=True` ensures past covariates are passed in `X_df`
            # which is needed for `hist_exog_list` in the API call.
            df_nixtla, x_df_nixtla = model_input.get_nixtla_input(include_past_covs=True)

            # Record the start time for performance measurement.
            start_time: float = time.time()

            # Determine the TimeGPT model to use based on the forecast horizon.
            timegpt_model_name: str = (
                TIMEGPT_SHORT_HORIZON_MODEL
                if model_input.forecast_horizon <= LONG_HORIZON_THRESHOLD
                else TIMEGPT_LONG_HORIZON_MODEL
            )

            # Configure finetuning parameters.
            finetune_steps_val: int = FINETUNE_STEPS_DEFAULT if finetune else 0
            finetune_depth_val: int = FINETUNE_DEPTH_DEFAULT if finetune else 0 # Only apply depth if finetuning is active

            # Call the Nixtla API's forecast endpoint.
            forecast_result_df: pd.DataFrame = self.client.forecast(
                df=df_nixtla,
                X_df=x_df_nixtla,  # Contains both past and future covariates (if provided)
                h=model_input.forecast_horizon,
                level=prediction_levels,
                hist_exog_list=model_input.past_covs,  # Specify which columns in X_df are historical exogenous
                model=timegpt_model_name,
                finetune_steps=finetune_steps_val,
                finetune_depth=finetune_depth_val,
            )

            # Calculate the total time taken for forecasting.
            forecast_time_seconds: float = time.time() - start_time

            # Prepare `ModelOutput` from the forecast result DataFrame.
            # The column names for point, lower, and upper bounds need to be constructed
            # as per NixtlaClient's output format.
            point_col: str = TIMEGPT_POINT_FORECAST_COL_PREFIX
            lower_col: str = f"{TIMEGPT_LOWER_BOUND_COL_PREFIX}-{prediction_levels[0]}"
            upper_col: str = f"{TIMEGPT_UPPER_BOUND_COL_PREFIX}-{prediction_levels[0]}"

            return ModelOutput(forecast_successful=True).from_df(
                forecast_df=forecast_result_df,
                column_names=(point_col, lower_col, upper_col),
                metadata={FORECAST_TIME_METADATA_KEY: forecast_time_seconds},
            )

        except Exception as e:
            # Catch any unexpected errors during the forecasting process.
            log_id: str = str(uuid.uuid4())
            logger.exception(
                f"Error in TimeGPT forecast (Log ID: {log_id}): {e}",
                extra={"log_id": log_id},
            )
            return ModelOutput(forecast_successful=False, error_log_id=log_id)


if __name__ == "__main__":
    # Example Usage: This block demonstrates how to use the TimeGPT class.
    # It creates a dummy DataFrame and ModelInput, then attempts a TimeGPT forecast.
    # This example requires the `NIXTLA_API_KEY` environment variable to be set.
    # Ensure you have `pandas` and `numpy` installed for this example.

    # Setup basic logging for the example script.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # IMPORTANT: Set your Nixtla API Key in your environment variables before running, e.g.:
    # export NIXTLA_API_KEY="YOUR_NIXTLA_API_KEY_HERE"
    # Or uncomment the following line for testing purposes (but do not commit real keys):
    # os.environ[NIXTLA_API_KEY_ENV_VAR] = "YOUR_TEST_API_KEY"

    # Create a dummy DataFrame for time series forecasting.
    # It must contain a 'date' column (or whatever is specified as the timestamp column in ModelInput).
    df_example = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'y1': np.arange(100),
        'y2': np.arange(50, 150),
        'future_cov1': np.arange(10, 110),
        'past_cov1': np.arange(5, 105),
    })
    # Ensure date column is in a format compatible with Nixtla (e.g., ISO 8601 string).
    df_example['date'] = df_example['date'].dt.strftime('%Y-%m-%d')

    # Create a ModelInput instance from the dummy data.
    example_model_input = ModelInput(
        df=df_example,
        targets=['y1', 'y2'],  # Example with multiple target series
        forecast_horizon=7,
        past_covs=['past_cov1'],
        future_covs=['future_cov1'],
        categorical_covs=[], # No categorical covariates in this example
        frequency='D',  # Daily frequency
        quantile=0.9,   # For 90% prediction intervals (5% lower, 95% upper)
    )

    timegpt_forecaster = TimeGPT()

    logger.info("Attempting TimeGPT forecast (short horizon, no finetuning)...")
    output_short_horizon = timegpt_forecaster.forecast(model_input=example_model_input, finetune=False)

    if output_short_horizon.forecast_successful:
        logger.info(f"TimeGPT (short horizon) forecast successful! Time: {output_short_horizon.metadata.get(FORECAST_TIME_METADATA_KEY, 'N/A'):.2f}s")
        logger.info("Point forecasts for 'y1':\n%s", output_short_horizon.point_forecast['y1'])
        if output_short_horizon.lower_bound is not None and output_short_horizon.upper_bound is not None:
            logger.info("Lower bounds for 'y1':\n%s", output_short_horizon.lower_bound['y1'])
            logger.info("Upper bounds for 'y1':\n%s", output_short_horizon.upper_bound['y1'])
    else:
        logger.error(f"TimeGPT (short horizon) forecast failed. Error Log ID: {output_short_horizon.error_log_id}")

    # Example with long horizon and finetuning
    long_horizon_input = ModelInput(
        df=df_example,
        targets=['y1'],
        forecast_horizon=70, # Exceeds LONG_HORIZON_THRESHOLD (60)
        past_covs=['past_cov1'],
        future_covs=['future_cov1'],
        categorical_covs=[],
        frequency='D',
        quantile=0.9,
    )
    logger.info("\nAttempting TimeGPT forecast (long horizon, with finetuning)...")
    output_long_horizon_finetuned = timegpt_forecaster.forecast(model_input=long_horizon_input, finetune=True)

    if output_long_horizon_finetuned.forecast_successful:
        logger.info(f"TimeGPT (long horizon, finetuned) forecast successful! Time: {output_long_horizon_finetuned.metadata.get(FORECAST_TIME_METADATA_KEY, 'N/A'):.2f}s")
        logger.info("Point forecasts for 'y1':\n%s", output_long_horizon_finetuned.point_forecast['y1'])
    else:
        logger.error(f"TimeGPT (long horizon, finetuned) forecast failed. Error Log ID: {output_long_horizon_finetuned.error_log_id}")