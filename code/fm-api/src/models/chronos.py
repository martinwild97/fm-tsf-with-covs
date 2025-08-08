###################################################################################################
# Not implemented anymore because of dependency issues (autogluon and uni2ts require different versions for gluonts)
###################################################################################################

from typing import Optional, Literal
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from src.models.foundation_model import FoundationModel
from src.api.schemas import ModelInput, Forecast
import pandas as pd
import numpy as np
import shutil
import torch


class Chronos(FoundationModel):
    """Handling of Chronos forecasts"""

    def __init__(self, model_path: Optional[str] = "chronos_base", **kwargs):
        super().__init__(model_path, **kwargs)

    def load_model(self, model_path: Optional[str] = None, **kwargs) -> None:
        """
        Loads a pre-trained Chronos model.

        Parameters
        ----------
        model_path
            Fast bolt chronos models: chronos_tiny, chronos_mini, chronos_small, chronos_base. Default: chronos_small
        kwargs
            Additional parameters for the model loading.
        """
        self.model = TimeSeriesPredictor()
        dummmy_ts_df = TimeSeriesDataFrame(
            [{"target": [0, 1, 2], "start": pd.Period("01-01-2000", freq="D")}]
        )
        self.model.fit(
            train_data=dummmy_ts_df, presets=model_path or self.model_path, verbosity=0
        )

    def forecast(self, input: ModelInput) -> Forecast:
        """Create forecast for given actuals.

        Parameters
        ----------
        input
            Input data object.

        Returns
        -------
        Forecast object with median, input.quantile & 1-input.quantile
        """

        # Possible regression models: “LR”, “GBM”, “CAT”, “XGB”, “RF”. Default: “LR”.
        cov_regressor = getattr(input, "cov_regressor", "LR")

        if input.actuals.multivariate:
            raise ValueError(
                "Multivariate time series are not supported by Chronos model."
            )

        # Create model conform autogluonts dataframe.
        autogluon_df = input.actuals.to_autogluon_df()

        # Add covariate regressor if covariates are present.
        if input.actuals.future_cov_names or input.actuals.past_cov_names:
            # Handle past covariates if provided by forecasting to forecast horizon.
            if input.actuals.past_cov_names:
                for past_cov in input.actuals.past_cov_names:
                    predictor = TimeSeriesPredictor(
                        prediction_length=input.forecast_horizon,
                        target=past_cov,
                        verbosity=0,
                        log_to_file=False,
                        cache_predictions=False,
                    ).fit(autogluon_df, presets=self.model_path)
                    forecast = predictor.predict(autogluon_df)
                    # If no future covariates are provided, fill in the missing values in the past covariate.
                    if not input.actuals.future_cov_names:
                        timestamp = autogluon_df.index.get_level_values("timestamp")
                        new_index = pd.date_range(
                            start=timestamp[-1],
                            periods=input.forecast_horizon + 1,
                            freq=input.actuals.frequency,
                        )[1:]
                        item_ids = autogluon_df.index.get_level_values(
                            "item_id"
                        ).unique()
                        new_multi_index = pd.MultiIndex.from_product(
                            [item_ids, new_index], names=["item_id", "timestamp"]
                        )
                        nan_rows = pd.DataFrame(
                            np.nan, index=new_multi_index, columns=autogluon_df.columns
                        )
                        autogluon_df = pd.concat([autogluon_df, nan_rows])
                    # Fill missing values in past covariate with forecasted values.
                    missing_indices = autogluon_df[autogluon_df[past_cov].isna()].index
                    autogluon_df.loc[missing_indices, past_cov] = forecast[
                        "mean"
                    ].values

            # Split data into train and test sets (for training covariate regressor).
            train_data, test_data = autogluon_df.train_test_split(
                input.forecast_horizon
            )

            # Define hyperparameters for Chronos model.
            hyperparameters = {
                "Chronos": {
                    "model_path": self.model_path,
                    "target_scaler": "standard",
                    "covariate_regressor": cov_regressor,
                },
            }

            # Fit TimeSeriesPredictor.
            predictor = TimeSeriesPredictor(
                prediction_length=input.forecast_horizon,
                quantile_levels=[1 - input.quantile, input.quantile],
                target="target",
                known_covariates_names=input.actuals.future_cov_names
                + input.actuals.past_cov_names,
                verbosity=0,
                log_to_file=False,
                cache_predictions=False,
            ).fit(
                train_data,
                hyperparameters=hyperparameters,
            )

            # Predict with covariate regressor.
            forecast = predictor.predict(
                train_data, known_covariates=test_data.drop(columns=["target"])
            )

        else:
            predictor = TimeSeriesPredictor(
                prediction_length=input.forecast_horizon,
                quantile_levels=[1 - input.quantile, input.quantile],
                target="target",
                verbosity=0,
                log_to_file=False,
                cache_predictions=False,
            ).fit(autogluon_df, presets=self.model_path)
            forecast = predictor.predict(autogluon_df)

        # Prepare the output Forecast object.
        data, lower, upper = {}, {}, {}
        timestamp = [str(i) for i in forecast.index.get_level_values(1)][
            : input.forecast_horizon
        ]
        for item_id in forecast.index.levels[0]:
            data[item_id] = forecast.loc[item_id]["mean"].tolist()
            lower[item_id] = forecast.loc[item_id][str(1 - input.quantile)].tolist()
            upper[item_id] = forecast.loc[item_id][str(input.quantile)].tolist()

        return Forecast(
            timestamp=timestamp,
            data=data,
            lower=lower,
            upper=upper,
        )
