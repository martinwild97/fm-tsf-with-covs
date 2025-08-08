import pytest
from src.models.timesfm import TimesFM
from src.api.schemas import ModelInput, Forecast
from src.utils.time_series_utils import TimeSeries


class TestTimesFM:
    def test_forecast___given_multivariate_time_series___raises_error(self):
        # Arrange
        timesfm = TimesFM(model_path="google/timesfm-1.0-200m-pytorch")
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
            data={"value1": list(range(9)), "value2": list(range(9))},
            multivariate=1,
        )
        input = ModelInput(forecast_horizon=4, context_length=9, actuals=ts)

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Multivariate time series are not supported by TimesFM model.",
        ):
            timesfm.forecast(input)

    def test_forecast___given_forecast_horizon_exceeds_model_horizon___raises_error(
        self,
    ):
        # Arrange
        timesfm = TimesFM(model_path="google/timesfm-1.0-200m-pytorch", horizon_len=5)
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
            data={"value": list(range(9))},
        )
        input = ModelInput(forecast_horizon=6, context_length=9, actuals=ts)

        # Act & Assert
        with pytest.raises(
            ValueError,
            match=f"Forecast horizon {input.forecast_horizon} exceeds model horizon "
            f"{timesfm.model.horizon_len}. Consider loading the model with a larger horizon.",
        ):
            timesfm.forecast(input)

    def test_forecast___given_invalid_quantile___raises_error(self):
        # Arrange
        timesfm = TimesFM(model_path="google/timesfm-1.0-200m-pytorch")
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
            data={"value": list(range(9))},
        )
        input = ModelInput(
            forecast_horizon=4, context_length=9, actuals=ts, quantile=0.95
        )

        # Act & Assert
        # Assuming SUPPORTED_QUANTILES is accessible or hardcoded in the test file
        supported_quantiles_str = str([i / 10.0 for i in range(1, 10)])
        with pytest.raises(
            ValueError,
            match=f"Invalid quantile. TimesFM only supports following quantile values {supported_quantiles_str}",
        ):
            timesfm.forecast(input)

    def test_forecast___given_single_time_series___creates_forecast(self):
        # Arrange
        timesfm = TimesFM(model_path="google/timesfm-1.0-200m-pytorch")
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
            data={"value": list(range(9))},
        )
        input = ModelInput(forecast_horizon=4, context_length=9, actuals=ts)

        # Act
        forecasts = timesfm.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-10-01 00:00:00"

    def test_forecast___given_multi_time_series___creates_forecasts(self):
        # Arrange
        timesfm = TimesFM(model_path="google/timesfm-1.0-200m-pytorch")
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
            data={"value1": list(range(9)), "value2": list(range(9))},
        )
        input = ModelInput(forecast_horizon=4, context_length=9, actuals=ts)

        # Act
        forecasts = timesfm.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value1"]) == input.forecast_horizon
        assert len(forecasts.data["value2"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-10-01 00:00:00"

    def test_forecast___given_future_covariate___creates_forecast(self):
        # Arrange
        timesfm = TimesFM(model_path="google/timesfm-1.0-200m-pytorch")
        ts = TimeSeries(
            timestamp=[f"2024-01-{i + 1}" for i in range(12)],
            data={"value": list(range(9)), "cov": list(range(12))},
            future_cov_names=["cov"],
            forecast_horizon=3,
        )
        input = ModelInput(forecast_horizon=3, context_length=9, actuals=ts)

        # Act
        forecasts = timesfm.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-01-10"

    def test_forecast___given_past_covariate___creates_forecast(self):
        # Arrange
        timesfm = TimesFM(model_path="google/timesfm-1.0-200m-pytorch")
        ts = TimeSeries(
            timestamp=[f"2024-01-0{i + 1}" for i in range(9)],
            data={"value": list(range(9)), "cov": list(range(9))},
            past_cov_names=["cov"],
            forecast_horizon=3,
        )
        input = ModelInput(
            forecast_horizon=3,
            context_length=9,
            actuals=ts,
        )

        # Act
        forecasts = timesfm.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-01-10 00:00:00"

    def test_forecast___given_both_covariate_types___creates_forecast(self):
        # Arrange
        timesfm = TimesFM(model_path="google/timesfm-1.0-200m-pytorch")
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
            actuals=ts,
        )

        # Act
        forecasts = timesfm.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-01-10"
        assert (
            forecasts.lower
            == "TimesFM does not support quantile forecasts in combination with covariates."
        )
        assert (
            forecasts.upper
            == "TimesFM does not support quantile forecasts in combination with covariates."
        )

    @pytest.mark.parametrize(
        "model_path", ["google/timesfm-1.0-200m-pytorch"]
    )  # , "google/timesfm-2.0-500m-pytorch"])
    def test_load_model___given_different_parameters___loads_model(self, model_path):
        # Arrange
        timesfm = TimesFM()
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
            data={"value": list(range(9))},
        )
        input = ModelInput(forecast_horizon=4, context_length=9, actuals=ts)

        # Act
        timesfm.load_model(model_path=model_path)
        forecasts = timesfm.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2024-10-01 00:00:00"
