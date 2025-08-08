###################################################################################################
# Not implemented anymore because of dependency issues (autogluon and uni2ts require different versions for gluonts)
###################################################################################################

import pytest
from src.models.chronos import Chronos
from src.api.schemas import ModelInput, Forecast
from src.utils.time_series_utils import TimeSeries


class TestChronos:
    def test_forecast___given_multivariate_time_series___raises_error(self):
        # Arrange
        chronos = Chronos()
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
            data={"value1": list(range(9)), "value2": list(range(9))},
            multivariate=1,
        )
        input = ModelInput(forecast_horizon=4, context_length=9, actuals=ts)

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Multivariate time series are not supported by Chronos model.",
        ):
            chronos.forecast(input=input)

    def test_forecast___given_single_time_series___creates_forecast(self):
        # Arrange
        chronos = Chronos()
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
            data={"value": list(range(9))},
        )
        input = ModelInput(forecast_horizon=4, context_length=9, actuals=ts)

        # Act
        forecasts = chronos.forecast(input=input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-10-01 00:00:00"

    def test_forecast___given_multi_time_series___creates_forecasts(self):
        # Arrange
        chronos = Chronos()
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
            data={"value1": list(range(9)), "value2": list(range(9))},
            item_id_names=["item1", "item2"],
        )
        input = ModelInput(forecast_horizon=4, context_length=9, actuals=ts)

        # Act
        forecasts = chronos.forecast(input=input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value1"]) == input.forecast_horizon
        assert len(forecasts.data["value2"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-10-01 00:00:00"

    def test_forecast___given_future_covariate___creates_forecast(self):
        # Arrange
        chronos = Chronos()
        ts = TimeSeries(
            timestamp=[f"2024-01-{i + 1}" for i in range(12)],
            data={"value": list(range(9)), "cov": list(range(12))},
            future_cov_names=["cov"],
            forecast_horizon=3,
        )
        input = ModelInput(
            forecast_horizon=3, context_length=9, actuals=ts, cov_regressor="LR"
        )

        # Act
        forecasts = chronos.forecast(input=input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-01-10 00:00:00"

    def test_forecast___given_past_covariate___creates_forecast(self):
        # Arrange
        chronos = Chronos()
        ts = TimeSeries(
            timestamp=[f"2024-01-0{i + 1}" for i in range(9)],
            data={"value": list(range(9)), "cov": list(range(9))},
            past_cov_names=["cov"],
            forecast_horizon=3,
        )
        input = ModelInput(
            forecast_horizon=3, context_length=9, actuals=ts, cov_regressor="LR"
        )

        # Act
        forecasts = chronos.forecast(input=input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-01-10 00:00:00"

    def test_forecast___given_both_covariate_types___creates_forecast(self):
        # Arrange
        chronos = Chronos()
        ts = TimeSeries(
            timestamp=[f"2024-01-{i + 1}" for i in range(12)],
            data={
                "value": list(range(9)),
                "cov1": list(range(12)),
                "cov2": list(range(9)),
            },
            future_cov_names=["cov1"],
            past_cov_names=["cov2"],
            forecast_horizon=3,
        )
        input = ModelInput(
            forecast_horizon=3, context_length=9, actuals=ts, cov_regressor="LR"
        )

        # Act
        forecasts = chronos.forecast(input=input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-01-10 00:00:00"

    @pytest.mark.parametrize(
        "model_type",
        [
            "chronos_mini",  #'chronos_small', 'chronos_base', only with GPU
            "bolt_tiny",
            "bolt_mini",
            "bolt_small",
            "bolt_base",
        ],
    )
    def test_load_model___given_model_type___loads_model(self, model_type):
        # Arrange
        chronos = Chronos()
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
            data={"value": list(range(9))},
        )
        input = ModelInput(forecast_horizon=4, context_length=9, actuals=ts)

        # Act
        chronos.load_model(model_type=model_type)
        forecasts = chronos.forecast(input=input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-10-01 00:00:00"
