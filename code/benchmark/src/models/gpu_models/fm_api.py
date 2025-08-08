import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
import requests

from src.models.schemas import ModelInput, ModelOutput

# Base URL for the forecasting API.
BASE_URL: str = "http://localhost:8080/"

# Initialize a logger specifically for forecasting operations.
logger: logging.Logger = logging.getLogger("Forecast")


class FoundationModelAPIClient:
    """
    A class for interacting with a local forecasting API to generate forecasts
    using various deep learning and foundation models.

    It manages model switching and the handling of forecast requests and
    responses, including basic error handling and retries.
    """

    def __init__(self) -> None:
        """
        Initializes the FoundationModelAPIClient class.

        This involves setting up the HTTP headers for API requests and defining
        the available models and their corresponding API paths.
        """
        self.headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }

        # Defines available models, categorized by their API endpoint type.
        # 'ttm', 'timesfm', 'moirai' are specific model paths.
        # 'dl_models' is a list of models accessed via a generic 'dl_models' endpoint.
        self.models: Dict[str, Union[str, List[str]]] = {
            "ttm": "ibm-granite/granite-timeseries-ttm-r2",
            "timesfm": "google/timesfm-2.0-500m-pytorch",
            "moirai": "Salesforce/moirai-moe-1.0-R-base",
            "dl_models": ["NHITS", "TFT", "NBEATSx", "TiDE"],
        }

        # Attributes to keep track of the currently active model on the API.
        self.current_model: Optional[str] = None
        self.current_model_path: Optional[str] = None
        self.current_kwargs: Optional[Dict[str, Any]] = None

    def _switch_model(
        self, model_name: str, model_path: Optional[str] = None, kwargs: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Switches the active forecasting model on the API.
        Includes a retry mechanism (up to 2 attempts) in case of initial failure.

        Parameters
        ----------
        model_name : str
            The name of the model to switch to (e.g., "ttm", "NHITS").
            Must be one of the keys in `self.models` or an item in `self.models["dl_models"]`.
        model_path : Optional[str], default=None
            The specific path for the model on the API if different from `model_name` itself.
            Required for top-level models (e.g., "ttm" uses its defined path).
            For models in `dl_models`, `model_name` is used as `model_path`.
        kwargs : Optional[Dict[str, Any]], default=None
            Additional keyword arguments to configure the model on the API.

        Returns
        -------
        Optional[str]
            A UUID string representing the error log ID if the model switch fails
            after all attempts. Returns None if the switch is successful.
        """
        log_id: str = str(uuid.uuid4())  # Unique ID for this operation's logging
        switch_url: str = BASE_URL + "model"

        model_type_for_api: str
        model_path_for_api: str

        # Determine the `model_type` and `model_path` for the API request.
        if model_name in self.models["dl_models"]:
            model_type_for_api = "dl_models"
            model_path_for_api = model_name  # For dl_models, model_name is the path
        elif model_name in self.models:
            model_type_for_api = model_name
            # Use provided model_path or default from self.models for top-level models.
            model_path_for_api = model_path or self.models[model_name]
        else:
            logger.critical(
                f"Invalid model specified: '{model_name}'. Must be one of "
                f"{list(self.models.keys()) + self.models['dl_models']}",
                extra={"log_id": log_id},
            )
            return log_id  # Indicate failure with the generated log_id

        # Ensure kwargs is a dictionary, defaulting to empty if None.
        request_kwargs: Dict[str, Any] = kwargs if kwargs is not None else {}

        request_data: Dict[str, Any] = {
            "model_type": model_type_for_api,
            "model_path": model_path_for_api,
            "kwargs": request_kwargs,
        }

        # Attempt to switch the model, with a retry mechanism.
        for attempt in range(2):  # Two attempts: original + one retry
            try:
                response: requests.Response = requests.post(
                    switch_url, headers=self.headers, json=request_data
                )

                if response.status_code == 200:
                    try:
                        res_json: Dict[str, Any] = response.json()
                        # Check specific success message from the API.
                        if "status" in res_json and res_json["status"].startswith("Active model updated"):
                            # Update internal state on successful switch.
                            self.current_model = model_type_for_api
                            self.current_model_path = model_path_for_api
                            self.current_kwargs = request_kwargs
                            logger.info(
                                f"Successfully switched to model: {model_type_for_api} - {model_path_for_api}",
                                extra={"log_id": ""}, # No specific log_id for success
                            )
                            return None  # Success
                        else:
                            logger.critical(
                                f"Model switch attempt {attempt + 1} received status 200 but unexpected content: "
                                f"{response.text} - Model: {model_type_for_api} - Path: {model_path_for_api}",
                                extra={"log_id": log_id},
                            )
                    except requests.exceptions.JSONDecodeError:
                        logger.critical(
                            f"Model switch attempt {attempt + 1} received status 200 but failed to decode JSON "
                            f"response: {response.text} - Model: {model_type_for_api} - Path: {model_path_for_api}",
                            extra={"log_id": log_id},
                        )
                else:
                    # Log non-200 status codes.
                    logger.critical(
                        f"Model switch attempt {attempt + 1} failed. Received status code {response.status_code} "
                        f"with response: {response.text} - Model: {model_type_for_api} - Path: {model_path_for_api}",
                        extra={"log_id": log_id},
                    )

                # If it's the first attempt and failed, wait before retrying.
                if attempt < 1:
                    time.sleep(5)  # Wait for 5 seconds before retry
            except Exception as e:
                # Catch general exceptions during the request (e.g., network issues).
                logger.critical(
                    f"Unexpected error while switching model {model_type_for_api} - {model_path_for_api} "
                    f"on attempt {attempt + 1}: {e}",
                    extra={"log_id": log_id},
                    exc_info=True,  # Log full traceback
                )
                if attempt < 1:
                    time.sleep(5)

        return log_id  # Return error log ID if all attempts fail

    def _forecast_request(self, request_data: Dict[str, Any]) -> Union[Dict[str, Any], str]:
        """
        Sends a forecasting request to the API's /forecast endpoint.

        Parameters
        ----------
        request_data : Dict[str, Any]
            The JSON request payload for forecasting.

        Returns
        -------
        Union[Dict[str, Any], str]
            The parsed JSON response containing the forecast data if successful.
            Returns a UUID string (error log ID) if the request fails or the response
            cannot be parsed.
        """
        log_id: str = str(uuid.uuid4())
        forecast_url: str = BASE_URL + "forecast"

        try:
            response: requests.Response = requests.post(
                forecast_url, headers=self.headers, json=request_data
            )
            if response.status_code == 200:
                try:
                    res_json: Dict[str, Any] = response.json()
                    return res_json
                except requests.exceptions.JSONDecodeError:
                    # Handle cases where response is 200 but not valid JSON.
                    logger.error(
                        f"Error parsing JSON response: {response.text}",
                        extra={"log_id": log_id},
                    )
                    return log_id
            else:
                # Handle non-200 HTTP status codes for forecasting.
                error_message: str = f"Forecasting failed with status code {response.status_code}."
                try:
                    # Attempt to extract more specific error details from JSON response.
                    error_details: Dict[str, Any] = response.json()
                    detail = error_details.get("detail")
                    if isinstance(detail, list) and detail:
                        extracted_error: str = detail[0].get("msg", "Unknown error detail")
                        error_message += f" Detail: {extracted_error}"
                    elif isinstance(detail, str):
                        error_message += f" Detail: {detail}"
                    else:
                        error_message += f" Response JSON: {error_details}"
                except requests.exceptions.JSONDecodeError:
                    # If the error response itself is not JSON.
                    error_message += f" Response was not valid JSON. Text: {response.text[:500]}..."

                logger.error(error_message, extra={"log_id": log_id})
                return log_id

        except Exception as e:
            # Catch unexpected errors during the request (e.g., network issues).
            logger.exception(
                f"Unexpected error during forecasting API call: {e}",
                extra={"log_id": log_id},
            )
            return log_id

    def _is_uuid(self, input_string: Union[str, Any]) -> bool:
        """
        Checks if a given string is a valid UUID.

        Parameters
        ----------
        input_string : Union[str, Any]
            The string to check.

        Returns
        -------
        bool
            True if the string is a valid UUID, False otherwise.
        """
        if not isinstance(input_string, str):
            return False
        try:
            uuid.UUID(input_string)
            return True
        except ValueError:
            return False

    def _convert_list_to_numpy_array(self, input_dict: Dict[str, List[Union[float, int]]]) -> Dict[str, np.ndarray]:
        """
        Converts a dictionary where values are lists of numbers to a dictionary
        with the same keys but values converted to NumPy arrays.

        Parameters
        ----------
        input_dict : Dict[str, List[Union[float, int]]]
            A dictionary where keys are strings and values are lists of numbers.

        Returns
        -------
        Dict[str, np.ndarray]
            A new dictionary where keys are strings and values are NumPy arrays.
        """
        output_dict: Dict[str, np.ndarray] = {}
        for key, value in input_dict.items():
            output_dict[key] = np.array(value)
        return output_dict

    def _model_changed(
        self, model_name: str, model_path: Optional[str], kwargs: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Checks if the currently active model configuration (model type, path, and kwargs)
        differs from the requested model configuration.

        Parameters
        ----------
        model_name : str
            The requested model name (e.g., "ttm", "NHITS").
        model_path : Optional[str]
            The requested model path (if applicable).
        kwargs : Optional[Dict[str, Any]]
            The requested additional arguments for the model.

        Returns
        -------
        bool
            True if the model configuration has changed, False otherwise.
        """
        # Normalize kwargs for comparison: treat None as empty dict.
        normalized_current_kwargs: Dict[str, Any] = self.current_kwargs if self.current_kwargs is not None else {}
        normalized_requested_kwargs: Dict[str, Any] = kwargs if kwargs is not None else {}

        # Handle 'dl_models' category specifically, where model_name acts as model_path.
        if model_name in self.models["dl_models"]:
            return (
                self.current_model != "dl_models"
                or self.current_model_path != model_name  # For DL models, model_name is the path
                or normalized_current_kwargs != normalized_requested_kwargs
            )
        else:
            # For other models, compare against stored model and its specific path.
            return (
                self.current_model != model_name
                or self.current_model_path != model_path
                or normalized_current_kwargs != normalized_requested_kwargs
            )

    def forecast(
        self,
        model_input: ModelInput,
        model: str,
        model_path: Optional[str] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> ModelOutput:
        """
        Generates a forecast using the specified model.
        This method handles model switching if the requested model is different
        from the currently active one on the API. It also includes retry logic
        for model mismatches detected after a forecast.

        Parameters
        ----------
        model_input : ModelInput
            An instance of `ModelInput` containing the time series data for forecasting.
        model : str
            The name of the model to use (e.g., "ttm", "NHITS").
        model_path : Optional[str], default=None
            The specific path for the model on the API if needed (e.g., for `ttm`, `timesfm`).
            If None, the default path from `self.models` is used.
        kwargs : Optional[Dict[str, Any]], default=None
            Additional keyword arguments to configure the model on the API.

        Returns
        -------
        ModelOutput
            A `ModelOutput` instance containing the forecast results, success status,
            and any relevant metadata or error information.
        """
        try:
            # Prepare the request data in the format expected by the API.
            request_data: Dict[str, Any] = model_input.get_fm_api_input()

            # Check if the requested model is different from the currently active one.
            if self._model_changed(model, model_path, kwargs):
                logger.info(f"Model change detected. Attempting to switch to {model}...")
                switch_model_response: Optional[str] = self._switch_model(model, model_path, kwargs)
                if switch_model_response:
                    # If model switch fails, return a failed ModelOutput.
                    logger.error(
                        f"Error switching model to {model} (Log ID: {switch_model_response}). Aborting forecast."
                    )
                    return ModelOutput(
                        forecast_successful=False, error_log_id=switch_model_response
                    )

            # Perform the initial forecasting request.
            forecast_response_raw: Union[Dict[str, Any], str] = self._forecast_request(request_data)

            # Check if the raw response is an error log ID (string UUID).
            if self._is_uuid(forecast_response_raw):
                # If forecast failed, return a failed ModelOutput.
                logger.error(
                    f"Forecasting request failed (Log ID: {forecast_response_raw}). Aborting forecast."
                )
                return ModelOutput(
                    forecast_successful=False, error_log_id=forecast_response_raw
                )
            else:
                # Cast to Dict[str, Any] as it's a successful response.
                forecast_response: Dict[str, Any] = forecast_response_raw  # type: ignore

                # --- Model Mismatch Check and Retry Logic ---
                metadata: Optional[Dict[str, Any]] = forecast_response.get("metadata")
                actual_model_path: Optional[str] = metadata.get("model") if metadata else None
                
                needs_retry: bool = False
                if self.current_model_path and actual_model_path and self.current_model_path not in actual_model_path:
                    # Mismatch detected: the API reported using a different model than the one
                    # that was supposed to be active (`self.current_model_path`).
                    needs_retry = True
                    log_id_retry: str = str(uuid.uuid4())
                    logger.warning(
                        f"Model mismatch detected! Expected: '{self.current_model_path}', "
                        f"but forecast used: '{actual_model_path}'. Attempting retry (Log ID: {log_id_retry}).",
                        extra={"log_id": log_id_retry},
                    )

                if needs_retry:
                    # Step 1: Attempt to re-switch the model to the intended one.
                    switch_retry_response: Optional[str] = self._switch_model(model, model_path, kwargs)

                    if switch_retry_response:
                        logger.error(
                            f"Model switch RETRY failed for '{self.current_model_path}' (Log ID: {switch_retry_response}). Aborting forecast."
                        )
                        return ModelOutput(
                            forecast_successful=False, error_log_id=switch_retry_response
                        )

                    # Step 2: Try forecasting again after the re-switch.
                    logger.info(
                        f"Retrying forecast after model switch for '{self.current_model_path}' (Log ID: {log_id_retry})"
                    )
                    forecast_response_retry_raw: Union[Dict[str, Any], str] = self._forecast_request(request_data)

                    # Step 3: Check the result of the *retry* forecast.
                    if self._is_uuid(forecast_response_retry_raw):
                        logger.error(
                            f"Forecast RETRY failed for '{self.current_model_path}' (Log ID: {forecast_response_retry_raw}). Aborting."
                        )
                        return ModelOutput(
                            forecast_successful=False, error_log_id=forecast_response_retry_raw
                        )

                    forecast_response_retry: Dict[str, Any] = forecast_response_retry_raw  # type: ignore

                    # Step 4: Re-check for model mismatch on the retry response.
                    metadata_retry: Optional[Dict[str, Any]] = forecast_response_retry.get("metadata")
                    actual_model_path_retry: Optional[str] = metadata_retry.get("model") if metadata_retry else None

                    if (
                        self.current_model_path
                        and actual_model_path_retry
                        and self.current_model_path not in actual_model_path_retry
                    ):
                        # Mismatch STILL persists after retry. This indicates a serious issue.
                        final_error_log_id: str = str(uuid.uuid4())
                        logger.error(
                            f"Persistent model mismatch after retry! Expected: '{self.current_model_path}', "
                            f"but second forecast still used: '{actual_model_path_retry}'. Aborting (Log ID: {final_error_log_id}).",
                            extra={"log_id": final_error_log_id},
                        )
                        return ModelOutput(
                            forecast_successful=False, error_log_id=final_error_log_id
                        )

                    # If we reach here, the retry was successful and model mismatch resolved.
                    forecast_response = forecast_response_retry

                # --- Process Successful Forecast Response ---
                point_forecast: Dict[str, np.ndarray] = self._convert_list_to_numpy_array(
                    forecast_response["data"]
                )

                # Check if probabilistic forecasts (lower/upper bounds) are present and valid.
                # The API returns 'lower'/'upper' as dicts, keyed by target names.
                # We check if the first target (as a representative) exists in the 'lower' dict.
                lower_bound: Optional[Dict[str, np.ndarray]] = None
                upper_bound: Optional[Dict[str, np.ndarray]] = None

                if (
                    isinstance(forecast_response.get("lower"), dict)
                    and model_input.targets[0] in forecast_response["lower"]
                    and isinstance(forecast_response.get("upper"), dict)
                    and model_input.targets[0] in forecast_response["upper"]
                ):
                    lower_bound = self._convert_list_to_numpy_array(
                        forecast_response["lower"]
                    )
                    upper_bound = self._convert_list_to_numpy_array(
                        forecast_response["upper"]
                    )
                else:
                    logger.info(
                        f"Probabilistic forecasts (lower/upper bounds) not available or incomplete for model '{model}'."
                    )

                return ModelOutput(
                    point_forecast=point_forecast,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    forecast_successful=True,
                    metadata=forecast_response.get("metadata", {}),  # Default to empty dict if 'metadata' is missing
                )

        except Exception as e:
            # Catch any unexpected errors during the overall forecast process.
            log_id: str = str(uuid.uuid4())
            logger.exception(
                f"Error processing forecast response for model '{model}' (Log ID: {log_id}): {e}",
                extra={"log_id": log_id},
            )
            return ModelOutput(forecast_successful=False, error_log_id=log_id)
