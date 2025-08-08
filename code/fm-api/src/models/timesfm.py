import pandas as pd
import timesfm
import torch
import typing

import src.models.foundation_model
import src.api.schemas

DEFAULT_MODEL_PATH = "google/timesfm-2.0-500m-pytorch"
MODEL_2_0_500M_NUM_LAYERS = 50
MODEL_2_0_500M_USE_POSITIONAL_EMBEDDING = False
DEFAULT_RANDOM_SEED = 1223
SUPPORTED_QUANTILES = [i / 10.0 for i in range(1, 10)]
COVARIATE_QUANTILE_ERROR_MESSAGE = (
    "TimesFM does not support quantile forecasts in combination with covariates."
)


class TimesFM(src.models.foundation_model.FoundationModel):
    """
    TimesFM class for time series forecasting using the TimesFM model.

    This class extends FoundationModel to provide a concrete implementation
    for loading the TimesFM model and generating forecasts.
    """

    def __init__(
        self,
        model_path: typing.Optional[str] = DEFAULT_MODEL_PATH,
        **kwargs: typing.Any,
    ):
        """
        Initializes the TimesFM model handler.

        Parameters
        ----------
        model_path
            Path to the pre-trained model directory or the name of a pre-trained
            model from the TimesFM model repository. Defaults to
            DEFAULT_MODEL_PATH.
        **kwargs
            Additional keyword arguments to pass to the base FoundationModel
            and the TimesFM model configuration.
        """
        self.device: str = "gpu" if torch.cuda.is_available() else "cpu"
        super().__init__(model_path, **kwargs)

    def load_model(
        self, model_path: typing.Optional[str] = None, **kwargs: typing.Any
    ) -> None:
        """
        Loads the TimesFM model.

        Sets the model path, updates keyword arguments, determines the covariate
        handling mode, and initializes the TimesFM model instance with
        appropriate hyperparameters and checkpoint based on the model path.

        Parameters
        ----------
        model_path
            Path to the pre-trained model directory or the name of a pre-trained
            model from the TimesFM model repository. If None, the model_path
            provided during initialization is used.
        **kwargs
            Additional parameters for model loading, which update the internal
            keyword arguments dictionary.
        """
        self.model_path: typing.Optional[str] = model_path or self.model_path

        self.kwargs.update(kwargs)
        self.kwargs["backend"] = self.device

        self.xreg_mode: str = self.kwargs.pop("xreg_mode", "xreg + timesfm")

        # Initialize the model with specific parameters for the 2.0-500m version.
        # These differ from the default parameters for the 1.0-200m model.
        if self.model_path == DEFAULT_MODEL_PATH:
            self.model: timesfm.TimesFm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    num_layers=MODEL_2_0_500M_NUM_LAYERS,
                    use_positional_embedding=MODEL_2_0_500M_USE_POSITIONAL_EMBEDDING,
                    **self.kwargs,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self.model_path
                ),
            )
        else:
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(**self.kwargs),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self.model_path
                ),
            )

    def forecast(
        self, input_data: src.api.schemas.ModelInput
    ) -> src.api.schemas.Forecast:
        """
        Generates a forecast using the loaded TimesFM model.

        Validates the input data for multivariate series, forecast horizon,
        and supported quantile values. Performs the forecast using the TimesFM
        model, handling covariates if present. Calculates and returns the
        point forecast and confidence intervals (if no covariates are used).

        Parameters
        ----------
        input_data
            Input data object containing time series data, covariates,
            and forecasting parameters.

        Returns
        -------
        src.api.schemas.Forecast
            Forecast object containing timestamps, forecasted data,
            confidence intervals, and metadata.

        Raises
        ------
        ValueError
            If the input contains multivariate time series, the forecast horizon
            exceeds the model's horizon, or an invalid quantile value is provided.
        """
        if input_data.actuals.multivariate:
            raise ValueError(
                "Multivariate time series are not supported by TimesFM model."
            )

        if self.model.horizon_len < input_data.forecast_horizon:
            raise ValueError(
                f"Forecast horizon {input_data.forecast_horizon} exceeds model horizon "
                f"{self.model.horizon_len}. Consider loading the model with a larger horizon."
            )

        if input_data.quantile not in SUPPORTED_QUANTILES:
            raise ValueError(
                f"Invalid quantile. TimesFM only supports following quantile values "
                f"{SUPPORTED_QUANTILES}"
            )

        torch.manual_seed(DEFAULT_RANDOM_SEED)

        forecast_data: typing.Dict[str, typing.List[float]] = {}
        lower_bound: typing.Union[str, typing.Dict[str, typing.List[float]]] = {}
        upper_bound: typing.Union[str, typing.Dict[str, typing.List[float]]] = {}

        targets: typing.List[typing.List[float]] = [
            value
            for key, value in input_data.actuals.data.items()
            if key in input_data.actuals.value_names
        ]

        freq_info: typing.Dict[str, typing.Any] = self._process_frequency(
            input_data.actuals.frequency
        )

        self._start_time_measurement()

        if input_data.actuals.future_cov_names or input_data.actuals.past_cov_names:
            # If past covariates are provided, forecast them first to extend
            # their length to the forecast horizon.
            if input_data.actuals.past_cov_names:
                for past_cov_name in input_data.actuals.past_cov_names:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=self.device == "gpu"):
                            past_cov_forecast, _ = self.model.forecast(
                                [input_data.actuals.data[past_cov_name]],
                                freq=[freq_info["group"]],
                            )
                    input_data.actuals.data[past_cov_name].extend(
                        past_cov_forecast[0][: input_data.forecast_horizon]
                    )

            numerical_covariates: typing.Dict[str, typing.List[typing.List[float]]] = {
                key: [value for _ in range(len(input_data.actuals.value_names))]
                for key, value in input_data.actuals.data.items()
                if key not in input_data.actuals.value_names
                and key not in input_data.actuals.categorical_cov_names
            }
            categorical_covariates: typing.Dict[str, typing.List[typing.Any]] = {
                key: [value for _ in range(len(input_data.actuals.value_names))]
                for key, value in input_data.actuals.data.items()
                if key in input_data.actuals.categorical_cov_names
            }

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.device == "gpu"):
                    point_forecast, prob_forecast = self.model.forecast_with_covariates(
                        inputs=targets,
                        dynamic_numerical_covariates=numerical_covariates,
                        dynamic_categorical_covariates=categorical_covariates,
                        # Static categorical covariates are used to differentiate between the target series.
                        static_categorical_covariates={
                            "value_names": input_data.actuals.value_names
                        },
                        freq=[freq_info["group"]] * len(input_data.actuals.value_names),
                        xreg_mode=self.xreg_mode,
                    )
            # TimesFM does not support quantile forecasts when covariates are used.
            lower_bound = COVARIATE_QUANTILE_ERROR_MESSAGE
            upper_bound = COVARIATE_QUANTILE_ERROR_MESSAGE

        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.device == "gpu"):
                    point_forecast, prob_forecast = self.model.forecast(
                        targets, freq=[freq_info["group"]] * len(targets)
                    )

            # TimesFM returns quantiles at 0.1, 0.2, ..., 0.9.
            # The index for the lower bound is 10 - (quantile * 10).
            # The index for the upper bound is quantile * 10.
            lower_quantile_index = 10 - int(input_data.quantile * 10)
            upper_quantile_index = int(input_data.quantile * 10)

            for i, target_name in enumerate(input_data.actuals.value_names):
                lower_bound[target_name] = prob_forecast[i][lower_quantile_index][
                    : input_data.forecast_horizon
                ]
                upper_bound[target_name] = prob_forecast[i][upper_quantile_index][
                    : input_data.forecast_horizon
                ]

        self._end_time_measurement()
        forecast_time: float = self._end_time_measurement()

        if input_data.actuals.future_cov_names:
            timestamps: pd.DatetimeIndex = input_data.actuals.timestamp[
                -input_data.forecast_horizon :
            ]
        else:
            timestamps = pd.date_range(
                start=input_data.actuals.timestamp[-1],
                periods=input_data.forecast_horizon + 1,
                freq=input_data.actuals.frequency,
            )[1:]  # Exclude the last timestamp of the actuals.
        for i, target_name in enumerate(input_data.actuals.value_names):
            forecast_data[target_name] = point_forecast[i][
                : input_data.forecast_horizon
            ]

        if self.device == "gpu":
            torch.cuda.empty_cache()

        return src.api.schemas.Forecast(
            timestamp=[str(t) for t in timestamps],
            data=forecast_data,
            lower=lower_bound,
            upper=upper_bound,
            metadata={
                "model": self.model_path,
                "forecast_time_seconds": forecast_time,
                "kwargs": self.kwargs,
                "xreg_mode": self.xreg_mode,
            },
        )
