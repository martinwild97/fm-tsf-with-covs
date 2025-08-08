from typing import Optional, Dict, Any, List
import os
import shutil

import pytorch_lightning.callbacks
import torch
import pytorch_lightning
import neuralforecast
import neuralforecast.models
import neuralforecast.losses.pytorch

from src.api.schemas import ModelInput, Forecast
from src.models.foundation_model import FoundationModel

DEFAULT_MODEL_NAME: str = "NHITS"
DEFAULT_EARLY_STOP_PATIENCE_STEPS: int = 10
DEFAULT_BATCH_SIZE: int = 16
VALIDATION_SIZE_MULTIPLIER: int = 2
RECOMMENDED_HORIZON_LIMIT: int = 128

# Constants for input size calculation based on forecast horizon.
INPUT_SIZE_MULTIPLIER_H_LE_16: int = 4
INPUT_SIZE_MULTIPLIER_H_LE_32: int = 3
INPUT_SIZE_MULTIPLIER_H_LE_64: int = 2
INPUT_SIZE_MINIMUM_H: int = 1

MODELS: Dict[str, Any] = {
    "NHITS": neuralforecast.models.NHITS,
    "TFT": neuralforecast.models.TFT,
    "TiDE": neuralforecast.models.TiDE,
    "NBEATSx": neuralforecast.models.NBEATSx,
}

REPRODUCIBILITY_SEED: int = 1223


class StopTrainingLightningCallback(pytorch_lightning.callbacks.Callback):
    """A PyTorch Lightning callback to stop training if an external flag is set."""

    def __init__(self, dl_model_instance):
        super().__init__()
        self.dl_model_instance = dl_model_instance

    def on_train_batch_start(
        self,
        trainer: pytorch_lightning.Trainer,
        pl_module: pytorch_lightning.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if getattr(self.dl_model_instance, "_should_stop_training", False):
            print(
                "StopTrainingLightningCallback: Stopping training due to external signal."
            )
            trainer.should_stop = True


class DLModels(FoundationModel):
    """Class for handling non-pretrained deep learning models.

    Currently supports NHITS, TFT, TiDE, and NBEATSx.

    Parameters
    ----------
    model_path : Optional[str], default="NHITS"
        Name of the model to be used. Supported models are listed in the
        `MODELS` dictionary.
    **kwargs : Any
        Additional keyword arguments to be passed to the model constructor.
    """

    def __init__(
        self, model_path: Optional[str] = DEFAULT_MODEL_NAME, **kwargs: Any
    ) -> None:
        """Initializes the DLModels class.

        Sets the device to CUDA if available and initializes the base class.

        Parameters
        ----------
        model_path : Optional[str], default="NHITS"
            Name of the model to be used. Supported models are listed in the
            `MODELS` dictionary.
        **kwargs : Any
            Additional keyword arguments to be passed to the model constructor.
        """
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._should_stop_training: bool = False
        super().__init__(model_path, **kwargs)

    def load_model(self, model_path: Optional[str] = None, **kwargs: Any) -> None:
        """Loads a pre-trained DL model.

        Sets the model name based on the provided `model_path`. If the
        model name is not supported, a `ValueError` is raised, and the
        default model is loaded.

        Parameters
        ----------
        model_path : Optional[str], default=None
            Name of the model to be used. Supported models: "NHITS", "TFT",
            "TiDE" and "NBEATSx".
        **kwargs : Any
            Additional parameters for model loading, which update the class's
            internal keyword arguments dictionary.

        Raises
        ------
        ValueError
            If the provided `model_path` is not a supported model name.
        """
        self.kwargs.update(kwargs)
        if model_path not in MODELS.keys():
            self.model_path: str = DEFAULT_MODEL_NAME
            raise ValueError(
                f"Model {model_path} not implemented. Loaded default model {DEFAULT_MODEL_NAME}."
            )
        else:
            self.model_path = model_path

    def forecast(self, input_data: ModelInput) -> Forecast:
        """Generates a forecast for the given input data using a deep learning model.

        Handles data conversion, model training, prediction, and result
        formatting.

        Parameters
        ----------
        input_data : ModelInput
            Input data object containing actual time series data and forecast
            parameters defined by the `ModelInput` schema.

        Returns
        -------
        Forecast
            Forecast object containing timestamps, predicted medians, lower,
            and upper quantiles, and metadata.

        Raises
        ------
        ValueError
            If the input time series is multivariate, as this is not supported
            by the current DL model implementation.
        TypeError
            If a type error occurs during GluonTS dataset conversion, potentially
            due to frequency inconsistencies.
        """
        if input_data.actuals.multivariate:
            raise ValueError("Multivariate time series are not supported by DL models.")

        torch.manual_seed(REPRODUCIBILITY_SEED)

        # Dictionaries to store forecast results per target variable.
        data_forecast: Dict[str, List[float]] = {}
        lower_quantile_forecast: Dict[str, List[float]] = {}
        upper_quantile_forecast: Dict[str, List[float]] = {}

        dataframe = input_data.actuals.to_df().reset_index()
        dataframe.rename(
            columns={"timestamp": "ds", "target": "y", "value_names": "unique_id"},
            inplace=True,
        )

        train_dataframe = dataframe[~dataframe["y"].isnull()]
        future_dataframe = dataframe[dataframe["y"].isnull()]

        if future_dataframe.empty:
            future_dataframe = None

        # Determine the input size for the model.
        input_size: Optional[int] = self.kwargs.pop("input_size", None)
        if not input_size:
            freq_info = self._process_frequency(input_data.actuals.frequency)
            seasonality = freq_info.get(
                "seasonality", 1
            )

            # Number of autoregressive inputs. Best results with multiple of forecast horizon.
            # Input size based on requested forecast horizon with memory optimization.
            # Forecast horizons over RECOMMENDED_HORIZON_LIMIT are not recommended for this setup.
            if input_data.forecast_horizon <= 16:
                input_size = INPUT_SIZE_MULTIPLIER_H_LE_16 * input_data.forecast_horizon
            elif input_data.forecast_horizon <= 32:
                input_size = INPUT_SIZE_MULTIPLIER_H_LE_32 * input_data.forecast_horizon
            elif input_data.forecast_horizon <= 64:
                input_size = INPUT_SIZE_MULTIPLIER_H_LE_64 * input_data.forecast_horizon
            elif input_data.forecast_horizon > RECOMMENDED_HORIZON_LIMIT:
                input_size = RECOMMENDED_HORIZON_LIMIT
                print(
                    f"Warning: Forecast horizon of {input_data.forecast_horizon} is not recommended for this setup."
                )
            else:
                input_size = INPUT_SIZE_MINIMUM_H * input_data.forecast_horizon

            # Ensure input size is at least one seasonality.
            if input_size < seasonality:
                input_size = seasonality

        early_stop_patience_steps: int = self.kwargs.pop(
            "early_stop_patience_steps", DEFAULT_EARLY_STOP_PATIENCE_STEPS
        )

        batch_size: int = self.kwargs.pop("batch_size", DEFAULT_BATCH_SIZE)

        stop_callback = StopTrainingLightningCallback(dl_model_instance=self)

        model = MODELS[self.model_path](
            h=input_data.forecast_horizon,
            input_size=input_size,
            early_stop_patience_steps=early_stop_patience_steps,
            batch_size=batch_size,
            loss=neuralforecast.losses.pytorch.MQLoss(
                level=[input_data.quantile * 100]
            ),
            futr_exog_list=input_data.actuals.future_cov_names,
            hist_exog_list=input_data.actuals.past_cov_names,
            precision="16-mixed" if self.device == "cuda" else "32",
            callbacks=[stop_callback],
            **self.kwargs,
        )

        self._start_time_measurement()

        nf = neuralforecast.NeuralForecast(
            models=[model], freq=input_data.actuals.frequency
        )
        nf.fit(
            df=train_dataframe,
            val_size=VALIDATION_SIZE_MULTIPLIER * input_data.forecast_horizon,
            verbose=False,
        )

        # Adjust future dataframe timestamps if future covariates exist.
        if input_data.actuals.future_cov_names and future_dataframe is not None:
            expected_timestamps = nf.make_future_dataframe()["ds"].to_list()
            actual_timestamps = future_dataframe["ds"].to_list()

            if expected_timestamps != actual_timestamps:
                if len(expected_timestamps) * len(
                    input_data.actuals.value_names
                ) == len(actual_timestamps):
                    future_dataframe["ds"] = expected_timestamps * len(
                        input_data.actuals.value_names
                    )
                elif len(expected_timestamps) == len(actual_timestamps):
                    future_dataframe["ds"] = expected_timestamps
                else:
                    print(
                        "Warning: Ambiguous future dataframe timestamps. Attempting alignment."
                    )

        model.to(self.device)
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                prediction_dataframe = nf.predict(futr_df=future_dataframe)

        forecast_time = self._end_time_measurement()

        for target_name in input_data.actuals.value_names:
            forecast_target_dataframe = prediction_dataframe[
                prediction_dataframe["unique_id"] == target_name
            ]
            # Columns are ordered: unique_id, ds, median, lower, upper
            data_forecast[target_name] = forecast_target_dataframe.iloc[:, -3].tolist()
            lower_quantile_forecast[target_name] = forecast_target_dataframe.iloc[
                :, -2
            ].tolist()
            upper_quantile_forecast[target_name] = forecast_target_dataframe.iloc[
                :, -1
            ].tolist()

        # Convert forecast timestamps to string format.
        # Extract only the timestamps corresponding to the forecast horizon.
        forecast_timestamps: List[str] = [
            str(timestamp)
            for timestamp in prediction_dataframe["ds"].tolist()[
                -input_data.forecast_horizon :
            ]
        ]

        # Attempt to get feature importances if the model supports it.
        try:
            feature_importances_data: Dict[str, float] = nf.models[
                0
            ].feature_importances()
        except AttributeError:
            feature_importances_data = {}
        except Exception as e:
            print(f"An error occurred while getting feature importances: {e}")
            feature_importances_data = {}

        if self.device == "cuda":
            torch.cuda.empty_cache()

        del prediction_dataframe
        del nf
        del model
        del dataframe
        del train_dataframe
        if future_dataframe is not None:
            del future_dataframe

        lightning_logs_path = "lightning_logs"
        if os.path.exists(lightning_logs_path):
            try:
                shutil.rmtree(lightning_logs_path)
            except OSError as e:
                print(f"Error removing directory {lightning_logs_path}: {e}")

        return Forecast(
            timestamp=forecast_timestamps,
            data=data_forecast,
            lower=lower_quantile_forecast,
            upper=upper_quantile_forecast,
            metadata={
                "model": f"{self.model_path} - input_size: {input_size}",
                "forecast_time_seconds": forecast_time,
                "model_kwargs": self.kwargs,
                "feature_importances": self._prep_dict_with_df_json(
                    feature_importances_data
                )
                if feature_importances_data
                else {},
            },
        )
