from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Literal, Optional
import os
import shutil
import gc
import torch
import asyncio

from src.api.schemas import ModelInput, Forecast

# from src.models.chronos import Chronos
from src.models.moirai import Moirai
from src.models.timesfm import TimesFM
from src.models.ttm import TinyTimeMixer
from src.models.dl_models import DLModels

DEFAULT_LOAD_TIMEOUT = 600  # 10 minutes
DEFAULT_FORECAST_TIMEOUT = 300  # 5 minutes

MODEL_CLASSES = {
    # Chronos is not included in the current implementation because of dependencies issues with gluonts and uni2ts.
    # "chronos": Chronos,
    "moirai": Moirai,
    "timesfm": TimesFM,
    "ttm": TinyTimeMixer,
    "dl_models": DLModels,
}

# Global variables to store the active model instance and its type.
active_model: Optional[object] = None
active_model_type: Optional[str] = None
# Global lock for controlling access to the model/GPU.
model_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("initializing app lifespan...")
    global active_model, active_model_type, model_lock
    # Load only TimesFM as the default active model at startup.
    try:
        async with asyncio.timeout(DEFAULT_LOAD_TIMEOUT):
            active_model = TimesFM()
            active_model_type = "timesfm"
            print(f"Loaded default model: {active_model_type}")
    except asyncio.TimeoutError:
        print("Loading default model timed out.")
        active_model = None
        active_model_type = None
    except Exception as e:
        print("Error loading default model:", e)
        active_model = None
        active_model_type = None

    yield
    print("Shutting down active model.")
    # Acquire lock during shutdown to avoid conflict.
    async with model_lock:
        print("Acquired lock for final model cleanup.")
        # Run deletion and cleanup in a thread to be safe, though usually fast.
        await asyncio.to_thread(gc.collect)
        if torch.cuda.is_available():
            await asyncio.to_thread(torch.cuda.empty_cache)
        # Deletion itself might need a thread if __del__ is complex, but often not needed like this.
        if active_model is not None:
            del active_model
        active_model = None
        active_model_type = None
        print("Model resources released.")

    # Clean up temp directories.
    if os.path.exists("lightning_logs"):
        await asyncio.to_thread(shutil.rmtree, "lightning_logs")
    if os.path.exists("AutogluonModels"):
        await asyncio.to_thread(shutil.rmtree, "AutogluonModels")
    print("Lifespan shutdown complete.")


app = FastAPI(lifespan=lifespan)


# Forecasting Endpoint
@app.post("/forecast", response_model=Forecast)
async def generate_forecast(input: ModelInput) -> Forecast:
    """
    Generate a forecast using the currently active model.
    Execution is serialized by a lock to ensure single GPU usage.
    The active model is set at startup and can be updated via the `/model/` endpoint.

    Parameters
    ----------
    input : ModelInput
        An object containing time series data and parameters required for forecasting.
        - `forecast_horizon` (int): Number of time steps to predict.
        - `actuals` (TimeSeries): The historical time series data containing timestamps and values.
        - `context_length` (Optional[int]): Number of past data points to use for forecasting (default is entire series).
        - `quantile` (Optional[float]): The quantile for uncertainty estimation (default: 0.9).
        - `**kwargs`: Additional model-specific parameters (e.g., `batch_size`, `patch_size`, `optimization`).

    Returns
    -------
    Forecast
        The forecasted values, including confidence intervals.
        - `timestamp` (List[str]): The predicted future timestamps.
        - `data` (Dict[str, List[float]]): Predicted values for each series.
        - `lower` (Dict[str, List[float]] | str): Lower confidence bound (1 - quantile).
        - `upper` (Dict[str, List[float]] | str): Upper confidence bound (quantile).

    Raises
    ------
    HTTPException (400)
        If no active model is loaded.
    HTTPException (500)
        If an error occurs during forecasting.

    Example
    -------
    ```bash
    curl -X POST "http://localhost:8080/forecast/" -H "Content-Type: application/json" -d '{
        "forecast_horizon": 3,
        "actuals": {
            "timestamp": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01"],
            "data": {
                "value": [1, 2, 3, 4, 5],
                "future_cov": [1, 2, 3, 4, 5, 6, 7, 8],
                "past_cov": [1, 2, 3, 4, 5]
            },
            "future_cov_names": ["future_cov"],
            "past_cov_names": ["past_cov"]
        }
    }'
    ```

    Notes
    -----
    - Future covariates must match the length of timestamps.
    - Past covariates and values must have a length of `timestamps - forecast_horizon`.
    - The `timestamp_format` is optional but recommended.
    - Multi-time series forecasting is supported, but multivariate forecasts with covariates are not.
    - The `optimization` parameter can be used to optimize `patch_size` and `context_length` via backtesting.
    """
    global active_model, model_lock

    if active_model is None:
        raise HTTPException(
            status_code=400,
            detail="No active model loaded. Please update the model first.",
        )

    if hasattr(active_model, "_should_stop_training"):
        active_model._should_stop_training = False

    try:
        async with asyncio.timeout(DEFAULT_FORECAST_TIMEOUT):
            # Acquire the lock to ensure that only one request is processed at a time
            async with model_lock:
                forecast = await asyncio.to_thread(active_model.forecast, input)
                return forecast

    except asyncio.TimeoutError:
        # Stop training if the model supports it e.g. TTM with fine-tuning.
        if hasattr(active_model, "_should_stop_training"):
            active_model._should_stop_training = True
        raise HTTPException(
            status_code=408,
            detail=f"Forecast operation timed out after {DEFAULT_FORECAST_TIMEOUT} seconds.",
        )
    except Exception as e:
        # Stop training if the model supports it e.g. TTM with fine-tuning.
        if hasattr(active_model, "_should_stop_training"):
            active_model._should_stop_training = True
        raise HTTPException(status_code=500, detail=str(e))


class ModelUpdateRequest(BaseModel):
    """
    Request body for updating the active model.

    Parameters
    ----------
    model_type :
        The type of model to update. Must be one of ["moirai", "timesfm", "ttm", "dl_models"].
    model_path :
        The path to the new model (optional). Provide it to use an other version than the default.
        Example: - moirai: "Salesforce/moirai-1.1-R-small", "Salesforce/moirai-moe-1.0-R-base" (default)
                 - timesfm: "google/timesfm-1.0-200m-pytorch", "google/timesfm-2.0-500m-pytorch" (default)
                 - ttm: "ibm-granite/granite-timeseries-ttm-r1", "ibm-granite/granite-timeseries-ttm-r2" (default)
                 - dl_models: "TFT", "TiDE", "NBEATSx", "NHITS" (default)
    kwargs :
        Additional keyword arguments for model configuration (optional).
    """

    model_type: Literal["moirai", "timesfm", "ttm", "dl_models"]
    model_path: Optional[str] = None
    kwargs: dict = Field(default_factory=dict)

    class Config:
        protected_namespaces = ()


class ModelUpdateResponse(BaseModel):
    """
    Response model for model update requests.

    Parameters
    ----------
    status :
        A message indicating the result of the model update request.
    """

    status: str


@app.post("/model", response_model=ModelUpdateResponse)
async def update_model(config: ModelUpdateRequest) -> ModelUpdateResponse:
    """
    Update the active model with a new configuration.

    This endpoint allows switching to a new model by providing the model type,
    an optional model path, and additional configuration parameters.

    Parameters
    ----------
    config :
        The request body containing model type, path, and additional parameters.

    Returns
    -------
    ModelUpdateResponse
        A message indicating whether the model was successfully updated or an error occurred.

    Raises
    ------
    HTTPException (400)
        If the provided model type is not implemented.
    HTTPException (500)
        If an error occurs during model initialization.
    """
    global active_model, active_model_type, model_lock

    model_type = config.model_type
    new_model_instance = None

    try:
        async with asyncio.timeout(DEFAULT_LOAD_TIMEOUT):
            # Acquire the lock to ensure that only one request is processed at a time
            async with model_lock:
                if active_model is not None:
                    del active_model
                    active_model = None
                    active_model_type = None
                    await asyncio.to_thread(gc.collect)
                    if torch.cuda.is_available():
                        await asyncio.to_thread(torch.cuda.empty_cache)

                # Clean up any temporary model directories.
                if os.path.exists("lightning_logs"):
                    await asyncio.to_thread(shutil.rmtree, "lightning_logs")
                if os.path.exists("AutogluonModels"):
                    await asyncio.to_thread(shutil.rmtree, "AutogluonModels")

                if model_type not in MODEL_CLASSES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model '{model_type}' is not implemented. "
                        f"Please choose from: {list(MODEL_CLASSES.keys())}",
                    )

                # --- Instantiate the new model IN A THREAD ---
                print(f"Instantiating new model: {model_type}...")
                model_class = MODEL_CLASSES[model_type]

                # Correct way: Pass the class constructor and its arguments to to_thread
                if config.model_path:
                    # - OLD LINE (Incorrect): Instantiates model first, passes instance to to_thread
                    # new_model_instance = await asyncio.to_thread(
                    #     MODEL_CLASSES[model_type](
                    #         model_path=config.model_path, **config.kwargs
                    #     )
                    # )
                    # + NEW LINE (Correct): Passes constructor and args separately to to_thread
                    new_model_instance = await asyncio.to_thread(
                        model_class,  # Pass the callable (the class constructor)
                        model_path=config.model_path,  # Pass args for the constructor
                        **config.kwargs,
                    )
                else:
                    # - OLD LINE (Incorrect): Instantiates model first, passes instance to to_thread
                    # new_model_instance = await asyncio.to_thread(
                    #     MODEL_CLASSES[model_type](**config.kwargs)
                    # )
                    # + NEW LINE (Correct): Passes constructor and args separately to to_thread
                    new_model_instance = await asyncio.to_thread(
                        model_class,  # Pass the callable (the class constructor)
                        **config.kwargs,  # Pass args for the constructor
                    )
                # import pdb
                # pdb.set_trace()
                # Instantiate the new model with the provided model_path and additional parameters.
                # if config.model_path:
                #     new_model_instance = await asyncio.to_thread(
                #         MODEL_CLASSES[model_type](
                #             model_path=config.model_path, **config.kwargs
                #         )
                #     )
                # else:
                #     new_model_instance = await asyncio.to_thread(
                #         MODEL_CLASSES[model_type](**config.kwargs)
                #     )
                active_model = new_model_instance
                active_model_type = model_type

                msg = f"Active model updated to '{model_type}' with path: {active_model.model_path} and parameters: {active_model.kwargs}"
                print(msg)
                return ModelUpdateResponse(status=msg)

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Model update operation timed out after {DEFAULT_LOAD_TIMEOUT} seconds.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ModelInfo(BaseModel):
    """
    Response model for retrieving the currently active model information.

    Parameters
    ----------
    model_type :
        The type of the currently active model.
    model_path :
        The file path of the currently active model.
    model_kwargs :
        The keyword arguments used to configure the model.
    """

    model_type: str
    model_path: str
    model_kwargs: dict

    class Config:
        protected_namespaces = ()


@app.get("/model", response_model=ModelInfo)
async def get_current_model() -> ModelInfo:
    """
    Get information on the currently active model.

    This endpoint retrieves details about the model that is currently in use, including
    its type, path, and configuration parameters.

    Returns
    -------
    ModelInfo
        The active model's details.

    Raises
    ------
    HTTPException (400)
        If no active model is loaded.
    HTTPException (500)
        If an error occurs while retrieving model information.
    """
    if active_model is None or active_model_type is None:
        raise HTTPException(status_code=400, detail="No active model loaded.")

    try:
        path = getattr(active_model, "model_path", "N/A")
        kwargs = getattr(active_model, "kwargs", {})
        return ModelInfo(
            model_type=active_model_type,
            model_path=path,
            model_kwargs=kwargs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
