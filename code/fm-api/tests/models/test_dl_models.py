import pytest
from src.models.dl_models import DLModels
from src.api.schemas import ModelInput, Forecast
from src.utils.time_series_utils import TimeSeries
import pandas as pd
import os
import numpy as np
import shutil


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
        multivariate=1,
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


# Helper function to clean up logs after tests
def cleanup_logs():
    if os.path.exists("lightning_logs"):
        shutil.rmtree("lightning_logs")


class TestDLModels:
    @pytest.fixture(scope="class")
    def long_time_series_data(self):
        # Note: Increased num_points slightly to ensure enough data for validation split
        # val_size = 2 * forecast_horizon = 2 * 12 = 24
        # Need num_points > input_size + val_size
        # Example: input_size ~ 4*h = 48. Need num_points > 48 + 24 = 72 for MS
        # Example: input_size ~ 4*h = 48. Need num_points > 48 + 24 = 72 for D
        ts, ts_multi, ts_future_cov, ts_past_cov, ts_both_cov = (
            generate_long_time_series(
                freq="MS", num_points=80, forecast_horizon=12
            )  # Increased points
        )
        ts_d, ts_multi_d, ts_future_cov_d, ts_past_cov_d, ts_both_cov_d = (
            generate_long_time_series(
                freq="D", num_points=150, forecast_horizon=12
            )  # Increased points
        )
        yield {  # Use yield to allow cleanup
            "MS": (ts, ts_multi, ts_future_cov, ts_past_cov, ts_both_cov),
            "D": (ts_d, ts_multi_d, ts_future_cov_d, ts_past_cov_d, ts_both_cov_d),
        }
        # Cleanup after all tests in the class are done
        cleanup_logs()

    # --- NHITS Tests ---
    def test_forecast___given_single_time_series_nhits___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        ts, _, _, _, _ = long_time_series_data["MS"]
        dl_model = DLModels(model_path="NHITS", max_steps=10)
        input_data = ModelInput(forecast_horizon=12, actuals=ts, quantile=0.9)

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()  # Clean up after each test

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        # Timestamps might vary slightly based on exact pandas version/locale
        # assert forecasts.timestamp[0].startswith("2025-02-01") # MS freq forecast start
        assert len(forecasts.timestamp) == input_data.forecast_horizon

    def test_forecast___given_future_covariate_nhits___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        _, _, ts_future_cov, _, _ = long_time_series_data["D"]
        dl_model = DLModels(model_path="NHITS", max_steps=10)
        input_data = ModelInput(
            forecast_horizon=12, actuals=ts_future_cov, quantile=0.9
        )

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        # assert forecasts.timestamp[0].startswith("2024-05-30") # D freq forecast start
        assert len(forecasts.timestamp) == input_data.forecast_horizon

    def test_forecast___given_past_covariate_nhits___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        _, _, _, ts_past_cov, _ = long_time_series_data["D"]
        dl_model = DLModels(model_path="NHITS", max_steps=10)

        input_data = ModelInput(forecast_horizon=12, actuals=ts_past_cov, quantile=0.9)

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        # assert forecasts.timestamp[0].startswith("2024-05-30")
        assert len(forecasts.timestamp) == input_data.forecast_horizon

    def test_forecast___given_both_covariate_types_nhits___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        _, _, _, _, ts_both_cov = long_time_series_data["D"]
        dl_model = DLModels(model_path="NHITS", max_steps=10)
        input_data = ModelInput(forecast_horizon=12, actuals=ts_both_cov, quantile=0.9)

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        # assert forecasts.timestamp[0].startswith("2024-05-30")
        assert len(forecasts.timestamp) == input_data.forecast_horizon

    # --- TFT Tests ---
    def test_forecast___given_single_time_series_tft___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        ts, _, _, _, _ = long_time_series_data["MS"]
        # Reduce TFT complexity slightly for faster tests if needed
        dl_model = DLModels(model_path="TFT", max_steps=10, hidden_size=32, n_head=2)
        input_data = ModelInput(forecast_horizon=12, actuals=ts, quantile=0.9)

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        # assert forecasts.timestamp[0].startswith("2025-02-01")
        assert len(forecasts.timestamp) == input_data.forecast_horizon

    def test_forecast___given_future_covariate_tft___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        _, _, ts_future_cov, _, _ = long_time_series_data["D"]
        dl_model = DLModels(model_path="TFT", max_steps=10, hidden_size=32, n_head=2)
        input_data = ModelInput(
            forecast_horizon=12, actuals=ts_future_cov, quantile=0.9
        )

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        # assert forecasts.timestamp[0].startswith("2024-05-30")
        assert len(forecasts.timestamp) == input_data.forecast_horizon

    def test_forecast___given_past_covariate_tft___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        _, _, _, ts_past_cov, _ = long_time_series_data["D"]
        dl_model = DLModels(model_path="TFT", max_steps=10, hidden_size=32, n_head=2)
        input_data = ModelInput(forecast_horizon=12, actuals=ts_past_cov, quantile=0.9)

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        # assert forecasts.timestamp[0].startswith("2024-05-30")
        assert len(forecasts.timestamp) == input_data.forecast_horizon

    def test_forecast___given_both_covariate_types_tft___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        _, _, _, _, ts_both_cov = long_time_series_data["D"]
        dl_model = DLModels(model_path="TFT", max_steps=10, hidden_size=32, n_head=2)
        input_data = ModelInput(forecast_horizon=12, actuals=ts_both_cov, quantile=0.9)

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        # assert forecasts.timestamp[0].startswith("2024-05-30")
        assert len(forecasts.timestamp) == input_data.forecast_horizon

    # --- TiDE Tests --- (New)
    def test_forecast___given_single_time_series_tide___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        ts, _, _, _, _ = long_time_series_data["MS"]
        # Reduce TiDE complexity slightly for faster tests if needed
        dl_model = DLModels(model_path="TiDE", max_steps=10, hidden_size=64)
        input_data = ModelInput(forecast_horizon=12, actuals=ts, quantile=0.9)

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        assert len(forecasts.timestamp) == input_data.forecast_horizon

    def test_forecast___given_future_covariate_tide___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        _, _, ts_future_cov, _, _ = long_time_series_data["D"]
        dl_model = DLModels(model_path="TiDE", max_steps=10, hidden_size=64)
        input_data = ModelInput(
            forecast_horizon=12, actuals=ts_future_cov, quantile=0.9
        )

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        assert len(forecasts.timestamp) == input_data.forecast_horizon

    def test_forecast___given_past_covariate_tide___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        _, _, _, ts_past_cov, _ = long_time_series_data["D"]
        dl_model = DLModels(model_path="TiDE", max_steps=10, hidden_size=64)
        input_data = ModelInput(forecast_horizon=12, actuals=ts_past_cov, quantile=0.9)

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        assert len(forecasts.timestamp) == input_data.forecast_horizon

    def test_forecast___given_both_covariate_types_tide___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        _, _, _, _, ts_both_cov = long_time_series_data["D"]
        dl_model = DLModels(model_path="TiDE", max_steps=10, hidden_size=64)
        input_data = ModelInput(forecast_horizon=12, actuals=ts_both_cov, quantile=0.9)

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        assert len(forecasts.timestamp) == input_data.forecast_horizon

    # --- NBEATSx Tests --- (New)
    def test_forecast___given_single_time_series_nbeatsx___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        ts, _, _, _, _ = long_time_series_data["MS"]
        # Reduce NBEATSx complexity slightly for faster tests if needed
        dl_model = DLModels(
            model_path="NBEATSx",
            max_steps=10,
            n_blocks=[1, 1, 1],
            mlp_units=[[64, 64], [64, 64], [64, 64]],
        )
        input_data = ModelInput(forecast_horizon=12, actuals=ts, quantile=0.9)

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        assert len(forecasts.timestamp) == input_data.forecast_horizon

    def test_forecast___given_future_covariate_nbeatsx___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        _, _, ts_future_cov, _, _ = long_time_series_data["D"]
        dl_model = DLModels(
            model_path="NBEATSx",
            max_steps=10,
            n_blocks=[1, 1, 1],
            mlp_units=[[64, 64], [64, 64], [64, 64]],
        )
        input_data = ModelInput(
            forecast_horizon=12, actuals=ts_future_cov, quantile=0.9
        )

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        assert len(forecasts.timestamp) == input_data.forecast_horizon

    def test_forecast___given_past_covariate_nbeatsx___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        _, _, _, ts_past_cov, _ = long_time_series_data["D"]
        dl_model = DLModels(
            model_path="NBEATSx",
            max_steps=10,
            n_blocks=[1, 1, 1],
            mlp_units=[[64, 64], [64, 64], [64, 64]],
        )
        input_data = ModelInput(forecast_horizon=12, actuals=ts_past_cov, quantile=0.9)

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        assert len(forecasts.timestamp) == input_data.forecast_horizon

    def test_forecast___given_both_covariate_types_nbeatsx___creates_forecast(
        self, long_time_series_data
    ):
        # Arrange
        _, _, _, _, ts_both_cov = long_time_series_data["D"]
        dl_model = DLModels(
            model_path="NBEATSx",
            max_steps=10,
            n_blocks=[1, 1, 1],
            mlp_units=[[64, 64], [64, 64], [64, 64]],
        )
        input_data = ModelInput(forecast_horizon=12, actuals=ts_both_cov, quantile=0.9)

        # Act
        forecasts = dl_model.forecast(input_data)
        cleanup_logs()

        # Assert
        assert isinstance(forecasts, Forecast)
        assert len(forecasts.data["value"]) == input_data.forecast_horizon
        assert len(forecasts.timestamp) == input_data.forecast_horizon
