import pytest
import pandas as pd
from src.models.ttm import TinyTimeMixer
from src.api.schemas import ModelInput, Forecast
from src.utils.time_series_utils import TimeSeries
import numpy as np


# Generate longer time series data for testing
def generate_long_time_series(freq="MS", num_points=60, forecast_horizon=12):
    num_cov_points = num_points + forecast_horizon
    if freq == "MS":
        start_date = "2020-01-01"
        timestamp = (
            pd.date_range(start=start_date, periods=num_points, freq=freq)
            .strftime("%Y-%m-%d")
            .tolist()
        )
        cov_timestamp = (
            pd.date_range(start=start_date, periods=num_cov_points, freq=freq)
            .strftime("%Y-%m-%d")
            .tolist()
        )
    else:  # Assuming daily frequency
        start_date = "2024-01-01"
        timestamp = (
            pd.date_range(start=start_date, periods=num_points, freq=freq)
            .strftime("%Y-%m-%d")
            .tolist()
        )
        cov_timestamp = (
            pd.date_range(start=start_date, periods=num_cov_points, freq=freq)
            .strftime("%Y-%m-%d")
            .tolist()
        )

    data = {
        "value": np.random.rand(num_points).tolist(),
        "value1": np.random.rand(num_points).tolist(),
        "value2": np.random.rand(num_points).tolist(),
        "cov": np.random.rand(num_cov_points).tolist(),
        "cov1": np.random.rand(num_cov_points).tolist(),
        "cov2": np.random.rand(num_points).tolist(),
    }

    ts = TimeSeries(
        timestamp=timestamp,
        data={"value": data["value"]},
        frequency=freq,
        forecast_horizon=forecast_horizon,
    )
    ts_multi = TimeSeries(
        timestamp=timestamp,
        data={"value1": data["value1"], "value2": data["value2"]},
        frequency=freq,
        forecast_horizon=forecast_horizon,
    )
    ts_future_cov = TimeSeries(
        timestamp=cov_timestamp,
        data={"value": data["value"], "cov": data["cov"]},
        future_cov_names=["cov"],
        forecast_horizon=forecast_horizon,
        frequency=freq,
    )
    ts_past_cov = TimeSeries(
        timestamp=timestamp,
        data={"value": data["value"], "cov2": data["cov2"]},
        past_cov_names=["cov2"],
        forecast_horizon=forecast_horizon,
        frequency=freq,
    )
    ts_both_cov = TimeSeries(
        timestamp=cov_timestamp,
        data={"value": data["value"], "cov1": data["cov1"], "cov2": data["cov2"]},
        future_cov_names=["cov1"],
        past_cov_names=["cov2"],
        forecast_horizon=forecast_horizon,
        frequency=freq,
    )

    return ts, ts_multi, ts_future_cov, ts_past_cov, ts_both_cov


class TestTinyTimeMixer:
    FORECAST_HORIZON = 12
    CONTEXT_LENGTH = 52

    @pytest.fixture(scope="class")
    def time_series_data(self):
        ts, ts_multi, ts_future_cov, ts_past_cov, ts_both_cov = (
            generate_long_time_series(freq="MS", num_points=60, forecast_horizon=12)
        )
        ts_d, ts_multi_d, ts_future_cov_d, ts_past_cov_d, ts_both_cov_d = (
            generate_long_time_series(freq="D", num_points=2048, forecast_horizon=12)
        )
        return {
            "MS": (ts, ts_multi, ts_future_cov, ts_past_cov, ts_both_cov),
            "D": (ts_d, ts_multi_d, ts_future_cov_d, ts_past_cov_d, ts_both_cov_d),
        }

    def test_forecast___given_single_time_series_ttm___creates_forecast(
        self, time_series_data
    ):
        # Arrange
        ts, _, _, _, _ = time_series_data["MS"]
        ttm_model = TinyTimeMixer()
        input = ModelInput(forecast_horizon=12, context_length=120, actuals=ts)

        # Act
        forecasts = ttm_model.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2025-01-01 00:00:00"

    def test_forecast___given_multi_time_series_ttm___creates_forecasts(
        self, time_series_data
    ):
        # Arrange
        _, ts_multi, _, _, _ = time_series_data["MS"]
        ttm_model = TinyTimeMixer()
        input = ModelInput(forecast_horizon=12, context_length=120, actuals=ts_multi)

        # Act
        forecasts = ttm_model.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value1"]) == input.forecast_horizon
        assert len(forecasts.data["value2"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2025-01-01 00:00:00"

    def test_forecast___given_future_covariate_ttm___creates_forecast(
        self, time_series_data
    ):
        # Arrange
        _, _, ts_future_cov, _, _ = time_series_data["D"]
        ttm_model = TinyTimeMixer()
        input = ModelInput(
            forecast_horizon=12, context_length=120, actuals=ts_future_cov
        )

        # Act
        forecasts = ttm_model.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2029-08-10 00:00:00"

    def test_forecast___given_past_covariate_ttm___creates_forecast(
        self, time_series_data
    ):
        # Arrange
        _, _, _, ts_past_cov, _ = time_series_data["D"]
        ttm_model = TinyTimeMixer()

        input = ModelInput(forecast_horizon=12, context_length=120, actuals=ts_past_cov)

        # Act
        forecasts = ttm_model.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2029-08-10 00:00:00"

    def test_forecast___given_both_covariate_types_ttm___creates_forecast(
        self, time_series_data
    ):
        # Arrange
        _, _, _, _, ts_both_cov = time_series_data["D"]
        ttm_model = TinyTimeMixer()
        input = ModelInput(forecast_horizon=12, actuals=ts_both_cov)

        # Act
        forecasts = ttm_model.forecast(input)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input.forecast_horizon
        assert forecasts.timestamp[0] == "2029-08-10 00:00:00"

    def test_forecast_finetune___given_single_series___creates_forecast(
        self, time_series_data
    ):
        # Arrange
        ts, _, _, _, _ = time_series_data["D"]
        # Instantiate with fine-tuning percentage
        ttm_model = TinyTimeMixer(fine_tune_perc=0.9)
        input_data = ModelInput(
            forecast_horizon=self.FORECAST_HORIZON,
            context_length=self.CONTEXT_LENGTH,
            actuals=ts,
        )

        # Act
        forecasts = ttm_model.forecast(input_data)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        assert forecasts.timestamp[0] == "2029-08-10 00:00:00"
        # Check metadata reflects fine-tuning was attempted (based on input kwargs)
        assert forecasts.metadata["kwargs"]["fine_tune_perc"] == 0.9
        # Could add check for fine-tuning specific keys if model adds them to metadata

    def test_forecast_finetune___given_multi_series___creates_forecasts(
        self, time_series_data
    ):
        # Arrange
        _, ts_multi, _, _, _ = time_series_data["D"]
        fine_tune_perc = 0.9  # Example percentage
        ttm_model = TinyTimeMixer(fine_tune_perc=fine_tune_perc)
        input_data = ModelInput(
            forecast_horizon=self.FORECAST_HORIZON,
            context_length=self.CONTEXT_LENGTH,
            actuals=ts_multi,
        )

        # Act
        forecasts = ttm_model.forecast(input_data)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value1"]) == input_data.forecast_horizon
        assert len(forecasts.data["value2"]) == input_data.forecast_horizon
        assert forecasts.timestamp[0] == "2029-08-10 00:00:00"
        assert forecasts.metadata["kwargs"]["fine_tune_perc"] == fine_tune_perc

    def test_forecast_finetune___given_both_covariates___creates_forecast(
        self, time_series_data
    ):
        # Arrange
        _, _, _, _, ts_both_cov = time_series_data["D"]
        fine_tune_perc = 0.9  # Example percentage
        # Pass fine-tuning specific hyperparams via kwargs if needed, e.g., epochs
        ttm_model = TinyTimeMixer(
            fine_tune_perc=fine_tune_perc, fine_tune_epochs=5
        )  # Short epochs for test speed
        input_data = ModelInput(
            forecast_horizon=self.FORECAST_HORIZON,
            context_length=self.CONTEXT_LENGTH,
            actuals=ts_both_cov,
        )

        # Act
        forecasts = ttm_model.forecast(input_data)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        assert forecasts.timestamp[0] == "2029-08-10 00:00:00"
        assert forecasts.metadata["kwargs"]["fine_tune_perc"] == fine_tune_perc
        assert forecasts.metadata["kwargs"]["fine_tune_epochs"] == 5

    def test_forecast_finetune_100perc___given_single_series___creates_forecast(
        self, time_series_data
    ):
        # Arrange
        ts, _, _, _, _ = time_series_data["D"]
        fine_tune_perc = 1  # Use '100%' for fine-tuning
        # The split logic (80/20 within the perc) means train=80%, valid=20%
        ttm_model = TinyTimeMixer(fine_tune_perc=fine_tune_perc, fine_tune_epochs=5)
        input_data = ModelInput(
            forecast_horizon=self.FORECAST_HORIZON,
            context_length=self.CONTEXT_LENGTH,
            actuals=ts,
        )

        # Act
        forecasts = ttm_model.forecast(input_data)

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        assert forecasts.timestamp[0] == "2029-08-10 00:00:00"
        assert forecasts.metadata["kwargs"]["fine_tune_perc"] == fine_tune_perc
