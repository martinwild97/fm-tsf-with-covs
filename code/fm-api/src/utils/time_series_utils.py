from typing import List, Dict, Optional, Literal
import pandas as pd
import gluonts.dataset.common
import gluonts.dataset.pandas

# chronos model is not implemented because of dependency issues
# import autogluon.timeseries
from pydantic import BaseModel, model_validator


class TimeSeries(BaseModel):
    """Time series definition.

    Can handle univariate and multivariate time series with past and future
    covariates.

    Parameters
    ----------
    timestamp : List[str]
        List of timestamps as strings.
    data : Dict[str, List[float]]
        Dictionary where keys are value names (including covariates) and
        values are lists of corresponding numerical data points.
    frequency : Optional[str], default=None
        Pandas frequency string. If None, the frequency will be inferred
        from the timestamps.
    past_cov_names : Optional[List[str]], default=[]
        List of past covariate names present in the data dictionary.
    future_cov_names : Optional[List[str]], default=[]
        List of future covariate names present in the data dictionary.
    categorical_cov_names : Optional[List[str]], default=[]
        List of categorical covariate names present in the data dictionary.
        These must also be included in either `past_cov_names` or
        `future_cov_names`.
    forecast_horizon : Optional[int], default=None
        Length of forecast steps. This must be provided when future covariates
        are included to correctly handle data lengths.
    multivariate : Optional[Literal[0, 1]], default=0
        Flag indicating whether the time series is multivariate (1) or a
        collection of multiple univariate time series (0).
    timestamp_format : Optional[str], default=None
        Format string for parsing the timestamp strings into datetime objects.
        If None, pandas will attempt to infer the format.

    Raises
    ------
    ValueError
        If timestamps are invalid, frequency cannot be inferred, specified
        covariates are not in the data, categorical covariates are not
        listed as past or future covariates, forecast horizon is missing
        with future covariates, data lengths are inconsistent, or
        multivariate is set with covariates.

    Examples
    --------
    >>> timestamps = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
    >>> data = {"value": [1.0, 2.0, 3.0, 4.0, 5.0]}
    >>> ts = TimeSeries(timestamp=timestamps, data=data, timestamp_format="%Y-%m-%d")
    >>> print(ts.frequency)
    D
    >>> df = ts.to_df()
    >>> print(df.head())
                value
    timestamp
    2020-01-01    1.0
    2020-01-02    2.0
    2020-01-03    3.0
    2020-01-04    4.0
    2020-01-05    5.0
    """

    timestamp: List[str]
    data: Dict[str, List[float]]
    frequency: Optional[str] = None
    past_cov_names: Optional[List[str]] = []
    future_cov_names: Optional[List[str]] = []
    categorical_cov_names: Optional[List[str]] = []
    forecast_horizon: Optional[int] = None
    multivariate: Optional[Literal[0, 1]] = 0
    timestamp_format: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_timestamps_and_lengths(cls, values: Dict) -> Dict:
        """Validates timestamp format, data keys, lengths, and multivariate settings.

        Parameters
        ----------
        values : Dict
            The dictionary of input values before Pydantic model validation.

        Returns
        -------
        Dict
            The validated and potentially updated dictionary of values.

        Raises
        ------
        ValueError
            If timestamps are invalid, frequency cannot be inferred, specified
            covariates are not in the data, categorical covariates are not
            listed as past or future covariates, forecast horizon is missing
            with future covariates, data lengths are inconsistent, or
            multivariate is set with covariates.
        """
        timestamp_list: List[str] = values.get("timestamp", [])
        timestamp_format: Optional[str] = values.get("timestamp_format")
        data_dict: Dict[str, List[float]] = values.get("data", {})
        past_cov_name_list: List[str] = values.get("past_cov_names", [])
        future_cov_name_list: List[str] = values.get("future_cov_names", [])
        categorical_cov_name_list: List[str] = values.get("categorical_cov_names", [])
        forecast_horizon_steps: int = values.get("forecast_horizon", 0)
        is_multivariate: Literal[0, 1] = values.get("multivariate", 0)

        # Validate timestamps.
        for i, timestamp_str in enumerate(timestamp_list):
            try:
                pd.to_datetime(timestamp_str, format=timestamp_format)
            except Exception as e:
                raise ValueError(f"Timestamp at position {i} is not a valid date. {e}")

        # If frequency is not provided, try to infer it.
        if not values.get("frequency"):
            try:
                datetime_index = pd.to_datetime(timestamp_list, format=timestamp_format)
            except Exception as e:
                raise ValueError(
                    f"Error converting timestamps for frequency inference: {e}"
                )

            inferred_frequency = pd.infer_freq(datetime_index)
            if not inferred_frequency:
                raise ValueError(
                    "Frequency cannot be inferred from the timestamps. Please provide the frequency."
                )
            values["frequency"] = inferred_frequency

        len_timestamp = len(timestamp_list)

        # Check if covariate names are available as keys in data dict.
        for name in past_cov_name_list + future_cov_name_list:
            if name not in data_dict:
                raise ValueError(
                    f"Covariate '{name}' is not available in data dictionary."
                )

        # Check if categorical covariates are in past or future covariates.
        all_covariate_names = past_cov_name_list + future_cov_name_list
        for name in categorical_cov_name_list:
            if name not in all_covariate_names:
                raise ValueError(
                    f"Categorical covariate '{name}' must be part of either 'past_cov_names' or 'future_cov_names'."
                )

        # Check if forecast_horizon is provided when future covariates are given.
        if future_cov_name_list:
            if not forecast_horizon_steps:
                raise ValueError(
                    "If future covariates are given, 'forecast_horizon' must also be provided."
                )

            # Check length of future covariates.
            for name in future_cov_name_list:
                if name in data_dict and len(data_dict[name]) != len_timestamp:
                    raise ValueError(
                        f"The length of future covariate '{name}' must be equal to the length of 'timestamp'."
                    )

            # Check length of past covariates (accounting for forecast horizon).
            for name in past_cov_name_list:
                if (
                    name in data_dict
                    and len(data_dict[name]) != len_timestamp - forecast_horizon_steps
                ):
                    raise ValueError(
                        f"If future covariates are given the length of past covariate '{name}' must be "
                        f"equal to the length of 'timestamp' - 'forecast_horizon."
                    )

            # Check length of all other values.
            value_names_with_future = [
                key
                for key in data_dict
                if key not in (future_cov_name_list + past_cov_name_list)
            ]
            for name in value_names_with_future:
                if (
                    name in data_dict
                    and len(data_dict[name]) != len_timestamp - forecast_horizon_steps
                ):
                    raise ValueError(
                        f"If future covariates are given the length of value '{name}' must be "
                        f"equal to the length of 'timestamp' - 'forecast_horizon'."
                    )

        else:
            # Check length of all other values.
            value_names_without_future = [
                key
                for key in data_dict
                if key not in (future_cov_name_list + past_cov_name_list)
            ]
            for name in value_names_without_future:
                if name in data_dict and len(data_dict[name]) != len_timestamp:
                    raise ValueError(
                        f"The length of value '{name}' must be equal to the length of 'timestamp'."
                    )
            # Check length of past covariates.
            for name in past_cov_name_list:
                if name in data_dict and len(data_dict[name]) != len_timestamp:
                    raise ValueError(
                        f"The length of past covariate '{name}' must be equal to the length of 'timestamp'."
                    )

        # Check if multivariate is set correctly.
        if is_multivariate and (past_cov_name_list or future_cov_name_list):
            raise ValueError("Multivariate forecasts with covariates is not supported")

        return values

    @property
    def value_names(self) -> List[str]:
        """Returns a sorted list of the target value names.

        Returns
        -------
        List[str]
            A sorted list of names for the non-covariate values in the data.
        """
        value_name_list = [
            key
            for key in self.data
            if key not in (self.future_cov_names + self.past_cov_names)
        ]
        value_name_list.sort()
        return value_name_list

    def to_df(self) -> pd.DataFrame:
        """Transforms the TimeSeries object into a pandas DataFrame.

        The resulting DataFrame has the timestamp as the index and includes
        all data columns (values and covariates).

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame representing the time series data.
        """
        # Create DataFrame from data dictionary.
        dataframe = pd.DataFrame(
            dict([(key, pd.Series(value)) for key, value in self.data.items()])
        )
        dataframe["timestamp"] = self.timestamp
        # Convert timestamp column to datetime.
        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"], format=self.timestamp_format
        )
        # Melt dataframe to long format for gluonts.
        dataframe_long = pd.melt(
            dataframe,
            id_vars=["timestamp"] + self.future_cov_names + self.past_cov_names,
            value_vars=self.value_names,
            var_name="value_names",
            value_name="target",
        )
        dataframe_long.set_index("timestamp", inplace=True)

        return dataframe_long

    def to_gluonts(self) -> gluonts.dataset.common.Dataset:
        """Transforms the TimeSeries object into a GluonTS Dataset.

        Returns
        -------
        gluonts.dataset.common.Dataset
            A GluonTS Dataset object.

        Raises
        ------
        ValueError
            If the frequency cannot be determined or is inconsistent, preventing
            dataset creation.
        """
        dataframe = self.to_df()
        try:
            # The future_length parameter in PandasDataset.from_long_dataframe expects the
            # prediction length, which corresponds to the forecast horizon.
            return gluonts.dataset.pandas.PandasDataset.from_long_dataframe(
                dataframe,
                item_id="value_names",
                past_feat_dynamic_real=self.past_cov_names,
                feat_dynamic_real=self.future_cov_names,
                future_length=self.forecast_horizon if self.future_cov_names else 0,
                freq=self.frequency,
            )
        except TypeError as e:
            # Catch specific TypeError related to frequency inference failure in GluonTS.
            if "'NoneType' object has no len()" in str(e):
                raise ValueError(
                    "Can't build dataset because of frequency inconsistency in timestamps. "
                    "Provide a pandas frequency."
                )
            # Re-raise other TypeErrors.
            raise e

    # chronos model is not implemented because of dependency issues
    # def to_autogluon_df(self) -> autogluon.timeseries.TimeSeriesDataFrame:
    #     """Transforms the TimeSeries object into an Autogluon TimeSeriesDataFrame."""
    #     if self.multivariate:
    #         raise ValueError(
    #             "Multivariate time series are not supported in AutoGluon/Chronos."
    #         )

    #     dataframe = self.to_df().reset_index()
    #     return autogluon.timeseries.TimeSeriesDataFrame(dataframe, id_column="value_names")
