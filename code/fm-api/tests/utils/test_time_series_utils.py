import pytest
from src.utils.time_series_utils import TimeSeries
from gluonts.dataset.common import Dataset

# from autogluon.timeseries import TimeSeriesDataFrame
import pandas as pd


class TestTimeSeries:
    def test_validate_timestamps___given_invalid_format___raises_value_error(self):
        # Arrange & Act & Assert
        with pytest.raises(Exception) as excinfo:
            TimeSeries(timestamp=["invalid-date"], data={"value": [1]})
        assert "Unknown datetime string format" in str(excinfo.value)

    def test_check_equal_length___given_non_matching_value_lengths___raises_value_error(
        self,
    ):
        # Arrange & Act & Assert
        with pytest.raises(
            ValueError,
            match="The length of value 'value' must be equal to the length of 'timestamp'",
        ):
            TimeSeries(
                timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
                data={"value": list(range(8))},  # One less value
            )

    def test_check_equal_length___given_non_matching_covariate_lengths___raises_value_error(
        self,
    ):
        # Arrange & Act & Assert
        with pytest.raises(
            ValueError,
            match="The length of future covariate 'cov' must be equal to the length of 'timestamp'",
        ):
            TimeSeries(
                timestamp=[f"2024-01-{i + 1}" for i in range(12)],
                data={"value": list(range(9)), "cov": list(range(9))},  # Should be 12
                future_cov_names=["cov"],
                forecast_horizon=3,
            )

    def test_check_equal_length___given_non_matching_value_lengths_covariate___raises_value_error(
        self,
    ):
        # Arrange & Act & Assert
        with pytest.raises(
            ValueError, match="If future covariates are given the length of value"
        ):
            TimeSeries(
                timestamp=[f"2024-01-{i + 1}" for i in range(12)],
                data={
                    "value": list(range(12)),  # Should be 9
                    "cov": list(range(12)),
                },
                future_cov_names=["cov"],
                forecast_horizon=3,
            )

    def test_check_equal_length___given_non_matching_past_covariate_lengths_covariate___raises_value_error(
        self,
    ):
        # Arrange & Act & Assert
        with pytest.raises(
            ValueError,
            match="If future covariates are given the length of past covariate",
        ):
            TimeSeries(
                timestamp=[f"2024-01-{i + 1}" for i in range(12)],
                data={
                    "value": list(range(9)),
                    "cov1": list(range(12)),
                    "cov2": list(range(12)),
                },  # Should be 9
                future_cov_names=["cov1"],
                past_cov_names=["cov2"],
                forecast_horizon=3,
            )

    def test_check_equal_length___given_non_matching_past_covariate_lengths___raises_value_error(
        self,
    ):
        # Arrange & Act & Assert
        with pytest.raises(
            ValueError,
            match="The length of past covariate 'cov' must be equal to the length of 'timestamp'",
        ):
            TimeSeries(
                timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
                data={"value": list(range(9)), "cov": list(range(8))},  # One less
                past_cov_names=["cov"],
            )

    def test_check_forecast_horizon___missing_forecast_horizon_for_covariates___raises_value_error(
        self,
    ):
        # Arrange & Act & Assert
        with pytest.raises(
            ValueError,
            match="If future covariates are given, 'forecast_horizon' must also be provided",
        ):
            TimeSeries(
                timestamp=[f"2024-01-{i + 1}" for i in range(9)],
                data={"value": list(range(9)), "cov": list(range(12))},
                future_cov_names=["cov"],
            )

    def test_multivariate_and_covariate___covariates_and_multivariate_given___raises_value_error(
        self,
    ):
        # Arrange & Act & Assert
        with pytest.raises(
            ValueError, match="Multivariate forecasts with covariates is not supported"
        ):
            TimeSeries(
                timestamp=[f"2024-01-{i + 1}" for i in range(12)],
                data={"value": list(range(9)), "cov": list(range(12))},
                future_cov_names=["cov"],
                forecast_horizon=3,
                multivariate=1,
            )

    def test_multivariate_and_covariate___past_covariates_and_multivariate_given___raises_value_error(
        self,
    ):
        # Arrange & Act & Assert
        with pytest.raises(
            ValueError, match="Multivariate forecasts with covariates is not supported"
        ):
            TimeSeries(
                timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
                data={"value": list(range(9)), "cov": list(range(9))},
                past_cov_names=["cov"],
                multivariate=1,
            )

    def test_to_gluonts___transforming_object___yields_valid_gluonts_dataset(self):
        # Arrange
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
            data={"value": list(range(9))},
        )

        # Act
        ds = ts.to_gluonts()

        # Assert
        assert isinstance(ds, Dataset)
        assert len(ds) == 1

    def test_to_gluonts___given_covariate___yields_valid_gluonts_dataset(self):
        # Arrange
        ts = TimeSeries(
            timestamp=[f"2024-01-{i + 1}" for i in range(12)],
            data={"value": list(range(9)), "cov": list(range(12))},
            future_cov_names=["cov"],
            forecast_horizon=3,
        )

        # Act
        ds = ts.to_gluonts()

        # Assert
        assert isinstance(ds, Dataset)
        assert len(ds) == 1
        assert ds.num_feat_dynamic_real == 1

    def test_to_gluonts___given_past_covariate___yields_valid_gluonts_dataset(self):
        # Arrange
        ts = TimeSeries(
            timestamp=[f"2024-01-{i + 1}" for i in range(9)],
            data={"value": list(range(9)), "cov": list(range(9))},
            past_cov_names=["cov"],
            forecast_horizon=3,
        )

        # Act
        ds = ts.to_gluonts()

        # Assert
        assert isinstance(ds, Dataset)
        assert len(ds) == 1
        assert ds.num_past_feat_dynamic_real == 1

    def test_to_gluonts___given_both_covariate_types___yields_valid_gluonts_dataset(
        self,
    ):
        # Arrange
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

        # Act
        ds = ts.to_gluonts()

        # Assert
        assert isinstance(ds, Dataset)
        assert len(ds) == 1
        assert ds.num_past_feat_dynamic_real == 1
        assert ds.num_feat_dynamic_real == 1

    def test_to_gluonts___multivariate___yields_valid_gluonts_dataset(self):
        # Arrange
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
            data={"value1": list(range(9)), "value2": list(range(9))},
        )

        # Act
        ds = ts.to_gluonts()

        # Assert
        assert isinstance(ds, Dataset)
        assert len(ds) == 2

    def test_to_gluonts___frequency_inconsistency_no_frequency_provided___yields_value_error(
        self,
    ):
        # Arrange
        with pytest.raises(
            ValueError,
            match="Frequency cannot be inferred from the timestamps. Please provide the frequency.",
        ):
            ts = TimeSeries(
                timestamp=[
                    "2024-01-16",
                    "2024-02-16",
                    "2024-03-16",
                    "2024-04-16",
                    "2024-05-16",
                ],
                data={"value": list(range(5))},
            )
            # Act
            ts.to_gluonts()

    def test_to_gluonts___frequency_inconsistency_freqency_provided___yields_valid_gluonts_dataset(
        self,
    ):
        # Arrange
        ts = TimeSeries(
            timestamp=[
                "2024-01-16",
                "2024-02-16",
                "2024-03-16",
                "2024-04-16",
                "2024-05-16",
            ],
            data={"value": list(range(5))},
            frequency="M",
        )
        # Act
        ds = ts.to_gluonts()
        # Assert
        assert isinstance(ds, Dataset)

    def test_to_df___no_frequency___yields_correct_dataframe(self):
        # Arrange
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(3)],
            data={"value": [1, 2, 3]},
            timestamp_format="%Y-%m-%d",
        )

        # Act
        df = ts.to_df()

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "target" in df.columns
        assert "value_names" in df.columns

    def test_to_df___with_frequency___yields_correct_dataframe(self):
        # Arrange
        ts = TimeSeries(
            timestamp=[f"2024-0{i + 1}-01" for i in range(3)],
            data={"value": [1, 2, 3]},
            frequency="MS",
            timestamp_format="%Y-%m-%d",
        )

        # Act
        df = ts.to_df()

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "target" in df.columns
        assert "value_names" in df.columns

    def test_to_df___with_frequency_inconsistency___yields_correct_dataframe(self):
        # Arrange
        ts = TimeSeries(
            timestamp=[
                "2024-01-16",
                "2024-02-16",
                "2024-03-16",
                "2024-04-16",
                "2024-05-16",
            ],
            data={"value": list(range(5))},
            frequency="M",
            timestamp_format="%Y-%m-%d",
        )

        # Act
        df = ts.to_df()
        # Assert
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "target" in df.columns
        assert "value_names" in df.columns

    # def test_to_autogluon_df___yields_correct_correct_autogluon_dataframe(self):
    #     # Arrange
    #     ts = TimeSeries(
    #         timestamp=[f"2024-0{i + 1}-01" for i in range(9)],
    #         data={"value": list(range(9))},
    #         frequency="MS",
    #         timestamp_format="%Y-%m-%d",
    #     )

    #     # Act
    #     adf = ts.to_autogluon_df()

    #     # Assert
    #     assert isinstance(adf, TimeSeriesDataFrame)
    #     assert len(adf) == 9
    #     assert "target" in adf.columns
