import pytest
from src.models.moirai import Moirai
from src.api.schemas import ModelInput, Forecast
from src.utils.time_series_utils import TimeSeries


class TestMoirai:
    def test_forecast___given_single_time_series___creates_forecast(self):
        # Arrange
        moirai = Moirai()
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
            data={"value": list(range(9))},
        )
        input = ModelInput(
            forecast_horizon=4,
            context_length=9,
            batch_size=32,
            patch_size=32,
            actuals=ts,
        )

        # Act
        forecasts = moirai.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-10-01 00:00:00"

    def test_forecast___given_multi_time_series___creates_forecasts(self):
        # Arrange
        moirai = Moirai()
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
            data={"value1": list(range(9)), "value2": list(range(9))},
        )
        input = ModelInput(
            forecast_horizon=4,
            context_length=9,
            batch_size=32,
            patch_size=32,
            actuals=ts,
        )

        # Act
        forecasts = moirai.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value1"]) == input.forecast_horizon
        assert len(forecasts.data["value2"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-10-01 00:00:00"

    def test_forecast___given_future_covariate___creates_forecast(self):
        # Arrange
        moirai = Moirai()
        ts = TimeSeries(
            timestamp=[f"2024-01-{i + 1}" for i in range(12)],
            data={"value": list(range(9)), "cov": list(range(12))},
            future_cov_names=["cov"],
            forecast_horizon=3,
        )
        input = ModelInput(
            forecast_horizon=3,
            context_length=9,
            batch_size=32,
            patch_size=32,
            actuals=ts,
        )

        # Act
        forecasts = moirai.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-01-10 00:00:00"

    def test_forecast___given_past_covariate___creates_forecast(self):
        # Arrange
        moirai = Moirai()
        ts = TimeSeries(
            timestamp=[f"2024-01-0{i + 1}" for i in range(9)],
            data={"value": list(range(9)), "cov": list(range(9))},
            past_cov_names=["cov"],
            forecast_horizon=3,
        )
        input = ModelInput(
            forecast_horizon=3,
            context_length=9,
            batch_size=32,
            patch_size=32,
            actuals=ts,
        )

        # Act
        forecasts = moirai.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-01-10 00:00:00"

    def test_forecast___given_both_covariate_types___creates_forecast(self):
        # Arrange
        moirai = Moirai()
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
            forecast_horizon=3,
            context_length=9,
            batch_size=32,
            patch_size=32,
            actuals=ts,
        )

        # Act
        forecasts = moirai.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-01-10 00:00:00"

    @pytest.mark.parametrize(
        "model_path",
        [
            "Salesforce/moirai-moe-1.0-R-small",
            "Salesforce/moirai-moe-1.0-R-base",
            "Salesforce/moirai-1.0-R-small",
            "Salesforce/moirai-1.1-R-small",
        ],
    )
    # 'Salesforce/moirai-1.0-R-base', 'Salesforce/moirai-1.1-R-base',
    # 'Salesforce/moirai-1.0-R-large', 'Salesforce/moirai-1.1-R-large'])

    def test_load_model___given_model_path___loads_model(self, model_path):
        # Arrange
        moirai = Moirai()
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
            data={"value": list(range(9))},
        )
        input = ModelInput(
            forecast_horizon=4,
            context_length=9,
            batch_size=32,
            patch_size=32,
            actuals=ts,
        )
        # Act
        moirai.load_model(model_path=model_path)
        forecasts = moirai.forecast(input)
        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-10-01 00:00:00"
