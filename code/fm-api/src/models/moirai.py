import torch
import pandas as pd
from typing import Optional, Dict, Any, List, Iterator

import gluonts.dataset.multivariate_grouper
import gluonts.time_feature
import gluonts.model.forecast
import uni2ts.model.moirai
from src.models.foundation_model import FoundationModel
import src.api.schemas

DEFAULT_MOIRAI_MODEL_PATH: str = "Salesforce/moirai-1.1-R-base"
DEFAULT_PATCH_SIZE: str = "auto"
DEFAULT_BATCH_SIZE_MOIRAI: int = 32
ALLOWED_PATCH_SIZES: List[int | str] = [8, 16, 32, 64, 128, "auto"]
DEFAULT_NUM_SAMPLES: int = 100
MAX_CONTEXT_LENGTH: int = 512

REPRODUCIBILITY_SEED: int = 1223


class Moirai(FoundationModel):
    """Handling of Moirai forecasts.

    This class extends the FoundationModel to specifically implement the
    loading and forecasting logic for Moirai models.

    Parameters
    ----------
    model_path : Optional[str], default="Salesforce/moirai-1.1-R-base"
        Path to the pre-trained Moirai model or a Hugging Face model identifier.
    **kwargs : Any
        Additional keyword arguments to be passed during model loading or
        forecasting.
    """

    def __init__(
        self, model_path: Optional[str] = DEFAULT_MOIRAI_MODEL_PATH, **kwargs: Any
    ) -> None:
        """Initializes the Moirai model handler.

        Sets the device (CUDA or CPU) and calls the base class constructor.

        Parameters
        ----------
        model_path : Optional[str], default="Salesforce/moirai-1.1-R-base"
            Path or identifier of the pre-trained Moirai model.
        **kwargs : Any
            Additional keyword arguments for model initialization.
        """
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(model_path, **kwargs)

    def load_model(self, model_path: Optional[str] = None, **kwargs: Any) -> None:
        """Loads a pre-trained Moirai model.

        Retrieves the model from a specified path or identifier, sets
        model-specific parameters like `patch_size` and `batch_size`,
        validates their values, and loads the model onto the appropriate device.

        Parameters
        ----------
        model_path : Optional[str], default=None
            Path to the pre-trained model or a Hugging Face model identifier.
            If None, the default model path is used.
        **kwargs : Any
            Additional parameters for model loading, which update the class's
            internal keyword arguments dictionary.

        Raises
        ------
        ValueError
            If `patch_size` or `batch_size` have invalid values.
        """
        self.kwargs.update(kwargs)
        # Use the provided model_path or the default/initialized model_path.
        self.model_path = model_path or self.model_path

        # Set model specific parameters, popping them from kwargs.
        self.patch_size: int | str = self.kwargs.pop("patch_size", DEFAULT_PATCH_SIZE)
        self.batch_size: int = self.kwargs.pop("batch_size", DEFAULT_BATCH_SIZE_MOIRAI)

        # Validate patch_size.
        if self.patch_size not in ALLOWED_PATCH_SIZES:
            raise ValueError(
                f"Invalid patch_size: {self.patch_size}. Allowed values: {ALLOWED_PATCH_SIZES}"
            )

        # Validate batch_size.
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(
                f"Invalid batch_size: {self.batch_size}. Must be an integer greater than 0."
            )

        # Load the pre-trained Moirai module and move it to the specified device.
        self.model: uni2ts.model.moirai.MoiraiModule = (
            uni2ts.model.moirai.MoiraiModule.from_pretrained(
                self.model_path,
                **self.kwargs,  # Pass remaining kwargs to from_pretrained
            ).to(self.device)
        )

    def forecast(
        self, input_data: src.api.schemas.ModelInput
    ) -> src.api.schemas.Forecast:
        """Creates a forecast for given actuals using the Moirai model.

        Converts input data to a GluonTS dataset, initializes the Moirai
        forecast object, generates predictions, and formats the results
        into a Forecast schema object.

        Parameters
        ----------
        input_data : src.api.schemas.ModelInput
            Input data object containing actual time series data and forecast
            parameters.

        Returns
        -------
        src.api.schemas.Forecast
            Forecast object containing timestamps, predicted medians, lower,
            and upper quantiles, and metadata.
        """
        # Create model conform GluonTS dataset.
        # Normalize the frequency string to a standard GluonTS format.
        input_data.actuals.frequency = gluonts.time_feature.norm_freq_str(
            input_data.actuals.frequency
        )
        dataset = input_data.actuals.to_gluonts()

        torch.manual_seed(REPRODUCIBILITY_SEED)

        # Determine target dimension for multivariate handling.
        # If it's a collection of time series (multivariate=False), target_dim is 1 per item.
        # If it's a single multivariate time series (multivariate=True), target_dim is the number of series.
        target_dimension: int = len(dataset) if input_data.actuals.multivariate else 1

        feat_dynamic_real_dimension: int = dataset.num_feat_dynamic_real
        past_feat_dynamic_real_dimension: int = dataset.num_past_feat_dynamic_real

        model_forecast = uni2ts.model.moirai.MoiraiForecast(
            module=self.model,
            prediction_length=input_data.forecast_horizon,
            context_length=input_data.context_length if input_data.context_length <=MAX_CONTEXT_LENGTH else MAX_CONTEXT_LENGTH,
            patch_size=self.patch_size,
            num_samples=DEFAULT_NUM_SAMPLES,
            target_dim=target_dimension,
            feat_dynamic_real_dim=feat_dynamic_real_dimension,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dimension,
        ).to(self.device)

        predictor = model_forecast.create_predictor(batch_size=self.batch_size)

        # Dictionaries to store forecast results per target variable.
        data_forecast: Dict[str, List[float]] = {}
        upper_quantile_forecast: Dict[str, List[float]] = {}
        lower_quantile_forecast: Dict[str, List[float]] = {}

        # Group the dataset for multivariate prediction if necessary.
        if input_data.actuals.multivariate:
            grouper = gluonts.dataset.multivariate_grouper.MultivariateGrouper(
                len(dataset)
            )
            dataset = grouper(dataset)

        self._start_time_measurement()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.device == "cuda"):
                forecasts_iterator: Iterator[gluonts.model.forecast.Forecast] = (
                    predictor.predict(dataset)
                )

        forecast_time_seconds: float = self._end_time_measurement()

        # Clean up GPU memory after prediction.
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Get the first forecast object to extract timestamps.
        try:
            first_forecast_result: gluonts.model.forecast.Forecast = next(
                forecasts_iterator
            )
        except StopIteration:
            # Handle case where predictor.predict returns an empty iterator.
            print(
                "Warning: Predictor returned an empty iterator. No forecasts generated."
            )
            # Return an empty Forecast object or handle as appropriate.
            return src.api.schemas.Forecast(
                timestamp=[],
                data={},
                lower={},
                upper={},
                metadata={
                    "model": self.model_path,
                    "forecast_time_seconds": forecast_time_seconds,
                    "model_kwargs": self.kwargs,
                },
            )

        # Extract timestamps from the index of the first forecast result.
        # Convert each timestamp to a string.
        timestamp_list: List[str] = [
            str(item.to_timestamp()) for item in first_forecast_result.index
        ]

        all_forecast_results: List[gluonts.model.forecast.Forecast] = [
            first_forecast_result
        ] + list(forecasts_iterator)

        for i, target_name in enumerate(input_data.actuals.value_names):
            if input_data.actuals.multivariate:
                # If multivariate, each forecast object represents one time series.
                # The forecast for the i-th time series is the i-th forecast object.
                if i < len(all_forecast_results):
                    single_forecast_result = all_forecast_results[i]
                else:
                    print(
                        f"Warning: Fewer forecast results ({len(all_forecast_results)}) than value names ({len(input_data.actuals.value_names)})."
                    )
                    break

            else:
                # If not multivariate, and there's only one time series (len(value_names) == 1),
                # the single forecast object contains the prediction for that series.
                # If there are multiple *univariate* time series, the predictor.predict
                # iterator yields one forecast object per time series in the input dataset.
                # The i-th forecast object corresponds to the i-th item in the input dataset,
                # which we assume corresponds to the i-th value name in this loop.
                if i < len(all_forecast_results):
                    single_forecast_result = all_forecast_results[i]
                else:
                    print(
                        f"Warning: Fewer forecast results ({len(all_forecast_results)}) than value names ({len(input_data.actuals.value_names)})."
                    )
                    break

            # Extract median, upper, and lower quantile values.
            data_forecast[target_name] = single_forecast_result.median.tolist()
            upper_quantile_forecast[target_name] = single_forecast_result.quantile_ts(
                input_data.quantile
            ).tolist()
            lower_quantile_forecast[target_name] = single_forecast_result.quantile_ts(
                1 - input_data.quantile
            ).tolist()

        return src.api.schemas.Forecast(
            timestamp=timestamp_list,
            data=data_forecast,
            upper=upper_quantile_forecast,
            lower=lower_quantile_forecast,
            metadata={
                "model": self.model_path,
                "forecast_time_seconds": forecast_time_seconds,
                "model_kwargs": self.kwargs,
            },
        )
