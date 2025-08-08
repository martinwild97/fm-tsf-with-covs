from typing import List, Dict, Optional, Any
import src.utils.time_series_utils
from pydantic import BaseModel, model_validator

DEFAULT_FORECAST_HORIZON: int = 8
DEFAULT_QUANTILE: float = 0.9


class Forecast(BaseModel):
    """Output definition for forecast results.

    Parameters
    ----------
    timestamp : List[str]
        List of timestamps as strings for the forecast period.
    data : Dict[str, List[float]]
        Dictionary where keys are value names and values are lists of
        forecasted median values.
    lower : Dict[str, List[float]] | str
        Lower quantile forecast results. Keys are value names, and values
        are lists of the lower quantile predictions. Can also be a string
        if quantile forecast is not applicable or available.
    upper : Dict[str, List[float]] | str
        Upper quantile forecast results. Keys are value names, and values
        are lists of the upper quantile predictions. Can also be a string
        if quantile forecast is not applicable or available.
    metadata : Optional[Dict], default=None
        Optional dictionary containing additional metadata about the forecast,
        such as model information or forecast time.
    """

    timestamp: List[str]
    data: Dict[str, List[float]]
    lower: Dict[str, List[float]] | str
    upper: Dict[str, List[float]] | str
    metadata: Optional[Dict[str, Any]] = None


class ModelInput(BaseModel):
    """Base input definition for time series forecasts.

    Defines the structure for input data provided to a forecasting model,
    including the time series data, forecast horizon, context length, and
    quantile level.

    Parameters
    ----------
    forecast_horizon : int
        The number of future steps to forecast.
    actuals : src.utils.time_series_utils.TimeSeries
        A TimeSeries object containing the historical and potentially future
        covariate data.
    context_length : Optional[int], default=None
        The number of historical data points to include as context for
        forecasting. If None, it is automatically calculated based on
        the length of `actuals` and the `forecast_horizon`.
    quantile : Optional[float], default=0.9
        The quantile level for probabilistic forecasting. Used to determine
        the upper and lower bounds of the prediction interval (e.g., 0.9
        for a 90% prediction interval).
    **kwargs : Any
        Model-specific parameters such as `batch_size`, `patch_size`,
        and `optimization`. These are allowed due to `Config.extra = "allow"`.

    Notes
    -----
    The `Config.extra = "allow"` setting permits additional arbitrary keyword
    arguments to be passed during initialization, which are accessible via the
    model instance's attributes or by accessing `__pydantic_extra__`.

    """

    forecast_horizon: int
    actuals: src.utils.time_series_utils.TimeSeries
    context_length: Optional[int] = None
    quantile: Optional[float] = DEFAULT_QUANTILE

    class Config:
        # Allow extra attributes (model-specific parameters)
        extra = "allow"

    @model_validator(mode="before")
    @classmethod
    def set_forecast_horizon_and_process_context_length(cls, values: Dict) -> Dict:
        """Sets default forecast horizon if missing in actuals and calculates context length if not provided.

        Ensures the `forecast_horizon` is present in the `actuals` TimeSeries
        object, using the `ModelInput`'s horizon if the `actuals` object
        does not specify one. Calculates `context_length` if it's not
        explicitly set in the input values.

        Parameters
        ----------
        values : Dict
            The dictionary of input values before Pydantic model validation.

        Returns
        -------
        Dict
            The validated and potentially updated dictionary of values, with
            `actuals` instantiated as a TimeSeries object (if it was a dict)
            and `context_length` set.
        """

        forecast_horizon: int = values.get("forecast_horizon", DEFAULT_FORECAST_HORIZON)
        actuals: Any = values.get("actuals")
        if actuals is None:
            raise ValueError("Missing 'actuals' data in the request payload.")

        # Ensure actuals is a TimeSeries instance and has forecast_horizon set.
        if isinstance(actuals, dict):
            if "forecast_horizon" not in actuals or actuals["forecast_horizon"] is None:
                actuals["forecast_horizon"] = forecast_horizon
            actuals = src.utils.time_series_utils.TimeSeries(**actuals)
        elif isinstance(actuals, src.utils.time_series_utils.TimeSeries):
            if getattr(actuals, "forecast_horizon", None) is None:
                actuals = actuals.model_copy(
                    update={"forecast_horizon": forecast_horizon}
                )
        else:
            # If actuals is neither a dict nor a TimeSeries instance, let Pydantic handle the validation error.
            pass

        values["actuals"] = actuals

        # Calculate context_length if it is not provided in the input.
        if values.get("context_length") is None:
            # If future covariates exist, context is up to the start of the forecast horizon.
            # Otherwise, the entire time series is context.
            if getattr(actuals, "future_cov_names", None):
                context_length = len(actuals.timestamp) - actuals.forecast_horizon
            else:
                context_length = len(actuals.timestamp)

            values["context_length"] = context_length

        return values
