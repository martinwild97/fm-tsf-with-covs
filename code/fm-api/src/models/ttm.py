import pandas as pd
import math
import tempfile
import os
import shutil
import typing

import transformers
import torch
import torch.optim
import torch.optim.lr_scheduler

import src.models.foundation_model
import src.api.schemas
import tsfm_public
import tsfm_public.toolkit.get_model
import tsfm_public.toolkit.callbacks

# Available pretrained models and their context and prediction lengths can be found here:
# https://github.com/ibm-granite/granite-tsfm/blob/main/tsfm_public/resources/model_paths_config/ttm.yaml
AVAILABLE_CONTEXT_PREDICTION_LENGTHS = {
    512: [48, 96, 192, 336, 720],
    1024: [96, 192, 336, 720],
    1536: [96, 192, 336, 720],
    52: [16],
    90: [30],
    180: [60],
    360: [60],
}

DEFAULT_TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
DEFAULT_TTM_CONTEXT_LENGTH = 512
DEFAULT_FINE_TUNE_PERC = 0
MIN_FINE_TUNE_PERC = 0
MAX_FINE_TUNE_PERC_CLAMP = 0.99
DEFAULT_BATCH_SIZE = 64
DEFAULT_FINE_TUNE_LR = 5e-4
DEFAULT_EARLY_STOPPING_PATIENCE = 10
DEFAULT_MAX_STEPS = 2000
MIN_VALID_SAMPLES_FALLBACK = 10
TRAIN_VALID_SPLIT_RATIO = (
    0.8  # 80% for training, 20% for validation within fine-tune data.
)
MAX_CONTEXT_FRACTION_FINETUNE = 0.33
MAX_CONTEXT_FRACTION_INFERENCE = (
    1.0  # Use full data length for context during inference if not fine-tuning.
)
DEFAULT_DECODER_MODE = "mix_channel"
DEFAULT_FCM_CONTEXT_LENGTH = 1
DEFAULT_FCM_USE_MIXER = True
DEFAULT_FCM_MIX_LAYERS = 2
DEFAULT_ENABLE_FORECAST_CHANNEL_MIXING = True
DEFAULT_FCM_PREPEND_PAST = True
DEFAULT_RANDOM_SEED = 1223
ERROR_SHORT_TIMESERIES_FORECAST = (
    "The timeseries is too short for generating a valid forecast. "
    "This happens mostly in combination with future covariates."
)
ERROR_SHORT_TIMESERIES_FINETUNE = (
    "The time series data is too short to perform fine-tuning with the current configuration "
    "(context length, prediction length, batch size, fine-tune percentage). "
    "Adjust these parameters or provide a longer time series."
)


class StopTrainingCallback(transformers.TrainerCallback):
    """A callback to stop training if an external flag is set."""

    def __init__(self, model_instance):
        # Store a reference to the TinyTimeMixer instance
        self.model_instance = model_instance

    def on_step_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        # Check the flag on the model instance at the beginning of each step
        if getattr(self.model_instance, "_should_stop_training", False):
            print("StopTrainingCallback: Stopping training due to external signal.")
            control.should_training_stop = True


class TinyTimeMixer(src.models.foundation_model.FoundationModel):
    """
    TinyTimeMixer class for time series forecasting using the TTM model.

    This class extends FoundationModel to provide a concrete implementation
    for loading and optionally fine-tuning the TTM model, and generating
    forecasts.
    """

    def __init__(
        self,
        model_path: typing.Optional[str] = DEFAULT_TTM_MODEL_PATH,
        **kwargs: typing.Any,
    ):
        """
        Initializes the TinyTimeMixer model handler.

        Parameters
        ----------
        model_path
            Path to the pre-trained model directory or the name of a pre-trained
            model. Defaults to DEFAULT_TTM_MODEL_PATH.
        **kwargs
            Additional keyword arguments to pass to the base FoundationModel
            and the TTM model configuration, including fine-tuning parameters.
        """
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._should_stop_training: bool = False  # Flag to control training stop
        super().__init__(model_path, **kwargs)

    def load_model(
        self, model_path: typing.Optional[str] = None, **kwargs: typing.Any
    ) -> None:
        """
        Stores model path and configuration.

        Actual model loading is deferred to the `forecast` method as it
        depends on data characteristics derived by the TimeSeriesPreprocessor.
        This method updates the internal keyword arguments and sets up
        fine-tuning parameters.

        Parameters
        ----------
        model_path
            Path to the pre-trained model directory or the name of a pre-trained
            model. If None, the model_path provided during initialization is used.
        **kwargs
            Additional parameters for model loading and fine-tuning. These
            update the internal keyword arguments dictionary.
        """
        self.kwargs.update(kwargs)
        self.model_path: typing.Optional[str] = model_path or self.model_path

        # Determine fine-tuning behavior based on the provided percentage.
        self.fine_tune_perc: float = self.kwargs.get(
            "fine_tune_perc", DEFAULT_FINE_TUNE_PERC
        )

        # Ensure fine-tune percentage is within a valid range [0, 0.99].
        if self.fine_tune_perc >= 1:
            self.fine_tune_perc = MAX_FINE_TUNE_PERC_CLAMP
        elif self.fine_tune_perc < 0:
            self.fine_tune_perc = MIN_FINE_TUNE_PERC

        self.perform_finetune: bool = self.fine_tune_perc > 0

        # Reset model state if path or key parameters change to force reload.
        self.model: typing.Optional[torch.nn.Module] = None
        self.tsp: typing.Optional[tsfm_public.TimeSeriesPreprocessor] = None
        self.model_loaded_config: typing.Dict[str, typing.Any] = {}

    def _determine_context_length(
        self, data_length: int, max_context_fraction: float, prediction_length: int
    ) -> int:
        """
        Determines a suitable context length based on data length and available models.

        Selects the largest available context length from AVAILABLE_CONTEXT_PREDICTION_LENGTHS
        that does not exceed the maximum allowable fraction of the total data length.
        If no such length exists, the smallest available context length is used.
        At the same Time it ensures that the context length is compatible with the
        prediction length.

        Parameters
        ----------
        data_length
            The total length of the time series data.
        max_context_fraction
            Maximum fraction of data length context should occupy (between 0 and 1).
        prediction_length
            The length of the prediction horizon.

        Returns
        -------
        int
            The selected context length from AVAILABLE_CONTEXT_LENGTHS.
        """
        max_allowable_context: int = int(data_length * max_context_fraction)
        possible_context_lengths = []
        for (
            context_length,
            prediction_lengths,
        ) in AVAILABLE_CONTEXT_PREDICTION_LENGTHS.items():
            if context_length <= max_allowable_context:
                for available_prediction_length in prediction_lengths:
                    if prediction_length <= available_prediction_length:
                        possible_context_lengths.append(context_length)
                        break

        if not possible_context_lengths:
            return 512  # Default context length if no suitable option is found.
        return max(possible_context_lengths)

    def _create_preprocessor(
        self, input_data: src.api.schemas.ModelInput
    ) -> tsfm_public.TimeSeriesPreprocessor:
        """
        Initializes and trains the TimeSeriesPreprocessor.

        Converts the input TimeSeries object to a pandas DataFrame and
        configures the TimeSeriesPreprocessor based on the data characteristics
        and specified context/prediction lengths.

        Parameters
        ----------
        input_data
            Input data object containing time series data and covariates.

        Returns
        -------
        tsfm_public.TimeSeriesPreprocessor
            An initialized and trained TimeSeriesPreprocessor instance.
        """
        df: pd.DataFrame = input_data.actuals.to_df().reset_index()

        # Use provided kwargs for context/prediction length, otherwise defaults.
        # Ensure prediction length matches forecast horizon for consistency.
        context_length: int = self.kwargs.get(
            "context_length", DEFAULT_TTM_CONTEXT_LENGTH
        )
        prediction_length: int = input_data.forecast_horizon

        tsp: tsfm_public.TimeSeriesPreprocessor = tsfm_public.TimeSeriesPreprocessor(
            timestamp_column="timestamp",
            id_columns=["value_names"],
            target_columns=["target"],
            conditional_columns=input_data.actuals.past_cov_names,
            control_columns=input_data.actuals.future_cov_names,
            context_length=context_length,
            prediction_length=prediction_length,
            freq=input_data.actuals.frequency,
            scaling=True,
            scaler_type="standard",
            encode_categorical=False,
        )
        tsp.train(df)
        return tsp

    def _load_or_get_model(
        self,
        tsp: tsfm_public.TimeSeriesPreprocessor,
        input_data: src.api.schemas.ModelInput,
    ) -> torch.nn.Module:
        """
        Loads or retrieves the TTM model instance.

        Loads the model using `tsfm_public.toolkit.get_model` if it hasn't been
        loaded yet or if the configuration based on model path, context/prediction
        lengths, input channels, and fine-tuning settings has changed.

        Parameters
        ----------
        tsp
            The trained TimeSeriesPreprocessor instance, used to get channel information.
        input_data
            Input data object, used for accessing forecast horizon if needed for config.

        Returns
        -------
        torch.nn.Module
            The loaded TTM model instance.
        """
        # Configuration representing the current state of the model we need.
        current_config: typing.Dict[str, typing.Any] = {
            "model_path": self.model_path,
            "context_length": tsp.context_length,
            "prediction_length": tsp.prediction_length,
            "num_input_channels": tsp.num_input_channels,
            "prediction_channel_indices": tsp.prediction_channel_indices,
            "perform_finetune": self.perform_finetune,
        }
        if self.perform_finetune:
            # Include fine-tuning specific args in the config check if fine-tuning is enabled.
            current_config.update(self._get_finetune_model_args(tsp))

        # Load or reload the model if it's not initialized or the required config changed.
        if (self.model is None) or (self.model_loaded_config != current_config):
            model_args: typing.Dict[str, typing.Any] = {
                "model_path": current_config["model_path"],
                "context_length": current_config["context_length"],
                "prediction_length": current_config["prediction_length"],
                "num_input_channels": current_config["num_input_channels"],
                # Always needed for the head layer size regardless of fine-tuning.
                "prediction_channel_indices": current_config[
                    "prediction_channel_indices"
                ],
            }

            # Add other general kwargs stored in self.kwargs, excluding those
            # already handled or specifically for fine-tuning training args.
            finetune_kwarg_keys = [
                "fine_tune_perc",
                "fine_tune_epochs",
                "fine_tune_lr",
                "early_stopping_patience",
                "batch_size",
                "max_steps",
                "decoder_mode",
                "fcm_context_length",
                "fcm_use_mixer",
                "fcm_mix_layers",
                "enable_forecast_channel_mixing",
                "fcm_prepend_past",
            ]
            general_kwargs = {
                k: v
                for k, v in self.kwargs.items()
                if k not in current_config and k not in finetune_kwarg_keys
            }
            model_args.update(general_kwargs)

            # Add fine-tuning specific parameters if fine-tuning is enabled.
            if self.perform_finetune:
                model_args.update(self._get_finetune_model_args(tsp))

            self.model = tsfm_public.toolkit.get_model(**model_args).to(self.device)
            self.model_loaded_config = current_config

        return self.model

    def _calculate_min_ft_data_length(
        self, tsp: tsfm_public.TimeSeriesPreprocessor
    ) -> int:
        """
        Calculates the minimum number of data points required for fine-tuning.

        The minimum length is determined by the batch size, minimum validation
        samples, context length, and prediction length to ensure enough data
        is available for both training and validation splits.

        Parameters
        ----------
        tsp
            The trained TimeSeriesPreprocessor instance.

        Returns
        -------
        int
            The minimum required data length for fine-tuning.
        """
        batch_size: int = self.kwargs.get("batch_size", DEFAULT_BATCH_SIZE)
        min_train_samples: int = batch_size
        min_valid_samples: int = max(MIN_VALID_SAMPLES_FALLBACK, batch_size // 4)
        total_min_samples: int = min_train_samples + min_valid_samples
        # The minimum data length must accommodate the context and prediction
        # length after accounting for the minimum required samples.
        min_data_len: int = (
            total_min_samples + tsp.context_length + tsp.prediction_length - 1
        )
        return min_data_len

    def _get_finetune_model_args(
        self, tsp: tsfm_public.TimeSeriesPreprocessor
    ) -> typing.Dict[str, typing.Any]:
        """
        Returns keyword arguments for get_model specific to fine-tuning with covariates.

        Includes default fine-tuning parameters inspired by the official TTM
        fine-tuning example, which can be overridden by providing them in kwargs
        during load_model. Args related to exogenous channels are only included
        if the preprocessor identifies such channels.

        Parameters
        ----------
        tsp
            The trained TimeSeriesPreprocessor instance, used to get exogenous channel information.

        Returns
        -------
        Dict[str, Any]
            A dictionary of keyword arguments for model initialization during fine-tuning.
        """
        ft_args: typing.Dict[str, typing.Any] = {
            "decoder_mode": self.kwargs.get("decoder_mode", DEFAULT_DECODER_MODE),
            "fcm_context_length": self.kwargs.get(
                "fcm_context_length", DEFAULT_FCM_CONTEXT_LENGTH
            ),
            "fcm_use_mixer": self.kwargs.get("fcm_use_mixer", DEFAULT_FCM_USE_MIXER),
            "fcm_mix_layers": self.kwargs.get("fcm_mix_layers", DEFAULT_FCM_MIX_LAYERS),
            "enable_forecast_channel_mixing": self.kwargs.get(
                "enable_forecast_channel_mixing", DEFAULT_ENABLE_FORECAST_CHANNEL_MIXING
            ),
            "fcm_prepend_past": self.kwargs.get(
                "fcm_prepend_past", DEFAULT_FCM_PREPEND_PAST
            ),
        }

        # Only include exogenous_channel_indices and related args if they exist.
        if tsp.exogenous_channel_indices:
            ft_args["exogenous_channel_indices"] = tsp.exogenous_channel_indices
        else:
            print(
                "Warning: Fine-tuning requested but no exogenous channels found by preprocessor. "
                "Disabling forecast channel mixing."
            )
            # Remove related args if no exogenous channels are present.
            # Also disable forecast channel mixing as it requires exogenous features.
            ft_args.pop("exogenous_channel_indices", None)
            ft_args.pop("fcm_context_length", None)
            ft_args.pop("fcm_use_mixer", None)
            ft_args.pop("fcm_mix_layers", None)
            ft_args.pop("fcm_prepend_past", None)
            ft_args["enable_forecast_channel_mixing"] = (
                False  # Explicitly disable if no exogenous channels
            )

        return ft_args

    def _fine_tune_model(
        self, df: pd.DataFrame, tsp: tsfm_public.TimeSeriesPreprocessor
    ) -> None:
        """
        Fine-tunes the TTM model using the provided dataframe and preprocessor.

        Splits the data for training and validation, freezes the model backbone,
        sets up the Hugging Face Trainer with optimizer, scheduler, and callbacks
        (including early stopping), runs the training process, and cleans up
        the temporary output directory.

        Parameters
        ----------
        df
            The pandas DataFrame containing the time series data for fine-tuning.
        tsp
            The trained TimeSeriesPreprocessor instance.
        """
        # Data Splitting: Use a fraction of the data for training and a small
        # validation set for early stopping.
        train_split_end: float = self.fine_tune_perc * TRAIN_VALID_SPLIT_RATIO
        val_split_end: float = self.fine_tune_perc

        split_params: typing.Dict[str, typing.List[float]] = {
            "train": [0, train_split_end],
            "valid": [train_split_end, val_split_end],
            # A test split is not directly used here for evaluation,
            # but get_datasets expects a full range split.
            "test": [val_split_end, 1.0],
        }

        # Create Datasets: Generate Hugging Face datasets using the preprocessor.
        train_dataset, valid_dataset, _ = tsfm_public.get_datasets(
            ts_preprocessor=tsp,
            dataset=df,
            split_config=split_params,
            use_frequency_token=True,
        )

        # Freeze Backbone: Prevent gradients from flowing through the main model layers.
        if self.model and self.model.backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        else:
            print(
                "Warning: Model or model backbone not found during fine-tuning. Skipping backbone freeze."
            )

        # Setup Training: Define hyperparameters and Trainer arguments.
        batch_size: int = self.kwargs.get("batch_size", DEFAULT_BATCH_SIZE)
        learning_rate: float = self.kwargs.get("fine_tune_lr", DEFAULT_FINE_TUNE_LR)
        early_stopping_patience: int = self.kwargs.get(
            "early_stopping_patience", DEFAULT_EARLY_STOPPING_PATIENCE
        )
        max_steps: int = self.kwargs.get("max_steps", DEFAULT_MAX_STEPS)

        output_dir: str = tempfile.mkdtemp()

        finetune_args: transformers.TrainingArguments = transformers.TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            do_eval=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            report_to=[],  # Disable reporting to external services.
            save_strategy="epoch",
            save_total_limit=1,  # Save only the best model.
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            max_steps=max_steps,
        )

        # Optimizer and Scheduler: Configure the optimization process.
        # Only optimize parameters that require gradients (i.e., the unfrozen layers).
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=max_steps,
        )

        # Callbacks: Add functionality like early stopping and tracking.
        callbacks: typing.List[transformers.TrainerCallback] = []
        callbacks.append(
            transformers.EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=0.0,
            )
        )
        # Optional callback for tracking training progress (e.g., printing logs).
        callbacks.append(tsfm_public.toolkit.callbacks.TrackingCallback())

        # Custom callback to stop training based on an external flag.
        callbacks.append(StopTrainingCallback(self))

        # Train: Initialize and run the Trainer.
        trainer = transformers.Trainer(
            model=self.model,
            args=finetune_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            callbacks=callbacks,
            optimizers=(optimizer, scheduler),
        )
        # Start training / fine-tuning the model.
        trainer.train()

        # Cleanup: Remove the temporary output directory.
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        # The best model is automatically loaded into self.model if
        # load_best_model_at_end is True in TrainingArguments.

    def forecast(
        self, input_data: src.api.schemas.ModelInput
    ) -> src.api.schemas.Forecast:
        """
        Generates a forecast using the TTM model.

        Initializes the preprocessor, loads or fine-tunes the model, performs
        inference using the forecasting pipeline, processes the results, and
        returns a Forecast object.

        Parameters
        ----------
        input_data
            Input data object containing time series data and covariates.

        Returns
        -------
        src.api.schemas.Forecast
            Forecast object with timestamps, forecasted data, and metadata.
            Note: TTM does not inherently provide quantile forecasts, so lower
            and upper bounds are empty dictionaries.

        Raises
        ------
        ValueError
            If the processed timeseries data is too short for generating a valid forecast.
        """
        # Setup and Preprocessing: Prepare data and determine context length.
        torch.manual_seed(DEFAULT_RANDOM_SEED)
        df: pd.DataFrame = input_data.actuals.to_df().reset_index()

        # Calculate data length considering whether future covariates are present.
        data_length: int = len(input_data.actuals.timestamp)
        # If future covariates are used, the actual historical data length available
        # for context/training is shorter by the forecast horizon.
        if input_data.actuals.future_cov_names:
            data_length -= input_data.forecast_horizon

        # Determine context length automatically if not specified.
        if self.kwargs.get("context_length") is None:
            # Use different max context fractions based on whether fine-tuning is enabled.
            max_context_fraction = (
                MAX_CONTEXT_FRACTION_FINETUNE
                if self.perform_finetune
                else MAX_CONTEXT_FRACTION_INFERENCE
            )
            self.kwargs["context_length"] = self._determine_context_length(
                data_length, max_context_fraction, input_data.forecast_horizon
            )

        # Initialize and train the preprocessor.
        self.tsp = self._create_preprocessor(input_data)

        # Load Model: Get the model instance, potentially reloading if config changed.
        self.model = self._load_or_get_model(self.tsp, input_data)

        self._start_time_measurement()

        # Fine-tuning (Optional): Perform fine-tuning if enabled.
        if self.perform_finetune:
            # Check if enough data is available for fine-tuning.
            min_required_length = self._calculate_min_ft_data_length(self.tsp)
            if data_length < min_required_length:
                raise ValueError(
                    f"Data length ({data_length}) is too short for fine-tuning "
                    f"with context length {self.tsp.context_length} and prediction length "
                    f"{self.tsp.prediction_length}. Minimum required length is {min_required_length}. "
                    "Consider increasing the number of data points or reducing "
                    "context/prediction lengths or batch size."
                )
            self._fine_tune_model(df, self.tsp)

        # Inference: Use the TimeSeriesForecastingPipeline to generate the forecast.
        pipeline = tsfm_public.TimeSeriesForecastingPipeline(
            model=self.model,
            feature_extractor=self.tsp,
            explode_forecasts=False,  # Keep forecasts as lists in the output DataFrame.
            device=self.device,
        )

        # Run inference on the entire dataframe; the pipeline handles windowing.
        forecast_df: pd.DataFrame = pipeline(df)

        forecast_time_in_seconds: float = self._end_time_measurement()

        # Process Results: Extract the forecast for the relevant timestamp.
        # Determine the timestamp corresponding to the start of the forecast horizon.
        if input_data.actuals.future_cov_names:
            # If future covariates are used, the inference includes the future period,
            # so the result timestamp is relative to the end of the actuals *before*
            # the forecast horizon starts.
            result_timestamp = df["timestamp"].iloc[-(1 + input_data.forecast_horizon)]
        else:
            # Without future covariates, inference typically stops at the last actual,
            # and the forecast starts immediately after. The result timestamp is the last actual.
            result_timestamp = df["timestamp"].iloc[-1]

        # Filter the forecast results to get the row corresponding to the forecast start.
        forecast_row = forecast_df[forecast_df["timestamp"] == result_timestamp]

        if forecast_row.empty:
            raise ValueError(ERROR_SHORT_TIMESERIES_FORECAST)

        # Extract the prediction data for each target variable.
        data: typing.Dict[str, typing.List[float]] = {}
        for target in input_data.actuals.value_names:
            # Extract the list of predictions for the target and truncate to the forecast horizon.
            target_predictions: typing.List[float] = (
                forecast_row[forecast_row["value_names"] == target]["target_prediction"]
                .values[0]
                .tolist()
            )
            data[target] = target_predictions[: input_data.forecast_horizon]

        # Generate the timestamps for the forecast horizon.
        # Start from the timestamp *after* result_timestamp for the forecast period.
        timestamp: pd.DatetimeIndex = pd.date_range(
            start=result_timestamp,
            periods=input_data.forecast_horizon + 1,
            freq=input_data.actuals.frequency,
        )[1:]  # Exclude the start timestamp itself.

        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Return the Forecast object. TTM does not provide confidence intervals by default.
        return src.api.schemas.Forecast(
            timestamp=[str(ts) for ts in timestamp],
            data=data,
            lower={},
            upper={},
            metadata={
                "model": self.model_path
                + f" - c{self.model.config.context_length} - p{self.model.config.prediction_length}",
                "forecast_time_seconds": forecast_time_in_seconds,
                "kwargs": self.kwargs,
            },
        )


if __name__ == "__main__":
    # Example usage demonstrating how to initialize the model and generate a forecast.
    from src.utils.time_series_utils import TimeSeries
    import pandas as pd
    import numpy as np

    # Create a sample TimeSeries object with multiple values and covariates.
    sample_ts_length = 1010
    sample_forecast_horizon = 10
    sample_timestamp = (
        pd.date_range(start="2000-01-01", periods=sample_ts_length, freq="D")
        .strftime("%Y-%m-%d")
        .tolist()
    )

    sample_data = {
        "value1": [
            i + np.random.rand()
            for i in range(sample_ts_length - sample_forecast_horizon)
        ],
        "value2": [
            i + np.random.rand()
            for i in range(sample_ts_length - sample_forecast_horizon)
        ],
        "past_cov1": [
            i + np.random.rand()
            for i in range(sample_ts_length - sample_forecast_horizon)
        ],
        "past_cov2": [
            i + np.random.rand()
            for i in range(sample_ts_length - sample_forecast_horizon)
        ],
        # Future covariates must extend into the forecast horizon.
        "future_cov1": [i + np.random.rand() for i in range(sample_ts_length)],
        "future_cov2": [i + np.random.rand() for i in range(sample_ts_length)],
    }

    ts = TimeSeries(
        timestamp=sample_timestamp,
        data=sample_data,
        frequency="D",
        value_names=["value1", "value2"],  # Explicitly list value columns
        past_cov_names=["past_cov1", "past_cov2"],
        future_cov_names=["future_cov1", "future_cov2"],
        forecast_horizon=sample_forecast_horizon,
    )

    input_data = src.api.schemas.ModelInput(
        forecast_horizon=sample_forecast_horizon,
        actuals=ts,
    )

    # Initialize and use the TinyTimeMixer model.
    # Set fine_tune_perc > 0 to enable fine-tuning.
    model = TinyTimeMixer(
        fine_tune_perc=0.1
    )  # Example fine-tuning on first 10% of data
    forecast = model.forecast(input_data)
    print(forecast)
