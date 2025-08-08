import logging
import time
import uuid
import warnings
from typing import List, Dict, Union

import numpy as np # Implicitly used for array operations in some cases
import pandas as pd # Implicitly used by StatsForecast DataFrame operations
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

from src.models.schemas import ModelInput, ModelOutput

# Suppress warnings from the statsforecast library to keep console output clean.
warnings.filterwarnings("ignore")

# Initialize a logger specifically for forecasting operations.
logger: logging.Logger = logging.getLogger("Forecast")


class BaselineModels:
    """
    A forecaster class that wraps statistical time series models from the `statsforecast` library,
    specifically AutoARIMA, AutoETS, and AutoTheta.

    This class provides a `forecast` method that takes a `ModelInput` instance,
    processes the input data, fits the specified models, generates forecasts,
    and returns the results as a list of `ModelOutput` instances.

    For AutoARIMA, both past and future covariates (if provided in `ModelInput`)
    are automatically included in the forecasting process.
    """

    def forecast(
        self, model_input: ModelInput, models: List[str] = None
    ) -> List[ModelOutput]:
        """
        Generates forecasts using a selection of statistical baseline models.

        Parameters
        ----------
        model_input : ModelInput
            An instance of `ModelInput` containing the time series data,
            forecast horizon, and other relevant metadata.
        models : Optional[List[str]], default=["AutoARIMA", "AutoETS", "AutoTheta"]
            A list of strings specifying which baseline models to use for forecasting.
            Valid options are "AutoARIMA", "AutoETS", and "AutoTheta".
            If None, all three are used by default.

        Returns
        -------
        List[ModelOutput]
            A list of `ModelOutput` instances, one for each model specified.
            Each `ModelOutput` contains the forecast results, success status,
            and any relevant metadata or error information.

        Raises
        ------
        Exception
            Catches any unexpected exceptions during the forecasting process,
            logs the error, and returns `ModelOutput` instances indicating failure.
        """
        if models is None:
            models = ["AutoARIMA", "AutoETS", "AutoTheta"]

        # Mapping of model names (strings) to their corresponding StatsForecast model classes.
        # This allows dynamic selection of models.
        model_class_map: Dict[str, Union[AutoARIMA, AutoETS, AutoTheta]] = {
            "AutoARIMA": AutoARIMA(),
            "AutoETS": AutoETS(),
            "AutoTheta": AutoTheta(),
        }

        # Filter and instantiate the selected models.
        # Only models present in `model_class_map` are considered.
        selected_stats_models: List[Union[AutoARIMA, AutoETS, AutoTheta]] = [
            model_class_map[model_name]
            for model_name in models
            if model_name in model_class_map
        ]

        # If no valid models are selected, return empty results early.
        if not selected_stats_models:
            logger.warning("No valid baseline models selected for forecasting.")
            return []

        try:
            # Prepare the input data for StatsForecast.
            # `get_nixtla_input()` returns a tuple: (DataFrame with target, DataFrame with covariates)
            df_nixtla, x_df_nixtla = model_input.get_nixtla_input()

            # Determine the confidence level for prediction intervals.
            # StatsForecast expects this as an integer percentage (e.g., 90 for 90%).
            prediction_level: int = int(model_input.quantile * 100)

            # Initialize the StatsForecast instance.
            # `n_jobs=-1` utilizes all available CPU cores for parallel processing.
            sf: StatsForecast = StatsForecast(
                df=df_nixtla,
                models=selected_stats_models,
                freq=model_input.frequency,
                n_jobs=-1,
            )

            # Record the start time for performance measurement.
            start_time: float = time.time()

            # Generate the forecasts.
            # `h` is the forecast horizon, `df` is the training data, `X_df` are future covariates,
            # and `level` specifies the confidence levels for intervals.
            forecast_df: pd.DataFrame = sf.forecast(
                h=model_input.forecast_horizon,
                df=df_nixtla,
                X_df=x_df_nixtla,
                level=[prediction_level],
            )

            # Calculate the total time taken for forecasting.
            forecast_time_seconds: float = time.time() - start_time

            # Reset the index of the forecast DataFrame to make `unique_id` and `ds` columns.
            forecast_df.reset_index(inplace=True)

            # Prepare `ModelOutput` for each forecasted model.
            results: List[ModelOutput] = []
            for model_instance in selected_stats_models:
                # `str(model_instance)` gives the model name (e.g., "AutoARIMA").
                model_name_str: str = str(model_instance)
                
                # Construct column names for point forecast, lower bound, and upper bound.
                # StatsForecast appends suffixes like "-lo-level" and "-hi-level".
                point_forecast_col: str = model_name_str
                lower_bound_col: str = f"{model_name_str}-lo-{prediction_level}"
                upper_bound_col: str = f"{model_name_str}-hi-{prediction_level}"

                # Create a ModelOutput instance, indicating success and adding metadata.
                results.append(
                    ModelOutput(forecast_successful=True).from_df(
                        forecast_df,
                        (point_forecast_col, lower_bound_col, upper_bound_col),
                        metadata={"forecast_time_seconds": forecast_time_seconds},
                    )
                )
            return results

        except Exception as e:
            # Catch any unexpected errors during the process.
            # Generate a unique ID for easier log tracing.
            log_id: str = str(uuid.uuid4())
            logger.exception(
                f"Error in BaselineModels forecast method (Log ID: {log_id}): {e}",
                extra={"log_id": log_id},
            )
            # Return a list of `ModelOutput` indicating failure for each attempted model.
            return [
                ModelOutput(forecast_successful=False, error_log_id=log_id)
                for _ in models
            ]