import numpy as np
from typing import Tuple


class Metrics:
    """
    A collection of static methods for calculating various time series forecasting metrics.
    These metrics include error measures like MASE, RMSSE, MSIS, WQL, CFE, and PIS.
    """

    @staticmethod
    def _calculate_naive_scale_mae(training_data: np.ndarray, seasonal_period: int = 1) -> float:
        """
        Calculates the scaling factor for MASE and MSIS based on the Mean Absolute Error (MAE)
        of a (seasonal) naive in-sample forecast.

        Parameters
        ----------
        training_data : np.ndarray
            Historical (training) time series data.
        seasonal_period : int, default=1
            The seasonal period (m). Use m=1 for a non-seasonal naive forecast.

        Returns
        -------
        float
            The calculated scaling factor. Returns 0.0 if there is not enough data
            or if all naive forecast errors are zero.
        """
        num_data_points: int = len(training_data)
        if num_data_points <= seasonal_period:
            # Not enough data points for at least one naive forecast.
            return 0.0

        # Calculate naive forecast errors: y_t - y_{t-m}
        naive_forecast_errors: np.ndarray = (
            training_data[seasonal_period:] - training_data[:-seasonal_period]
        )

        if naive_forecast_errors.size == 0:
            return 0.0

        scale: float = np.mean(np.abs(naive_forecast_errors))

        # If scale is 0 (e.g., training data is constant or perfectly periodic),
        # this will lead to division by zero in MASE/MSIS, which needs to be handled
        # by the calling metric functions.
        return float(scale)

    @staticmethod
    def _calculate_naive_scale_rmse(training_data: np.ndarray, seasonal_period: int = 1) -> float:
        """
        Calculates the scaling factor for RMSSE based on the Root Mean Squared Error (RMSE)
        of a (seasonal) naive in-sample forecast.

        Parameters
        ----------
        training_data : np.ndarray
            Historical (training) time series data.
        seasonal_period : int, default=1
            The seasonal period (m). Use m=1 for a non-seasonal naive forecast.

        Returns
        -------
        float
            The calculated scaling factor. Returns 0.0 if there is not enough data
            or if all naive forecast errors are zero.
        """
        num_data_points: int = len(training_data)
        if num_data_points <= seasonal_period:
            return 0.0

        naive_forecast_errors: np.ndarray = (
            training_data[seasonal_period:] - training_data[:-seasonal_period]
        )

        if naive_forecast_errors.size == 0:
            return 0.0

        scale: float = np.sqrt(np.mean(naive_forecast_errors**2))
        return float(scale)

    @staticmethod
    def mase(
        y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonal_period: int = 1
    ) -> float:
        """
        Calculates the Mean Absolute Scaled Error (MASE).

        MASE measures the accuracy of forecasts relative to the MAE of a naive forecast
        on the training data. A value less than 1 indicates better performance than
        the naive forecast.

        Parameters
        ----------
        y_true : np.ndarray
            Array of true values for the forecast period.
        y_pred : np.ndarray
            Array of predicted values for the forecast period.
        y_train : np.ndarray
            Array of training data used to calculate the naive forecast scale.
        seasonal_period : int, default=1
            The seasonal period (m) for scaling. Use m=1 for non-seasonal data.

        Returns
        -------
        float
            The MASE value. Returns 0.0 if `forecast_mae` is 0, and np.inf if the
            scaling factor is 0 but `forecast_mae` is non-zero.
        """
        y_true_arr: np.ndarray = np.asarray(y_true)
        y_pred_arr: np.ndarray = np.asarray(y_pred)
        y_train_arr: np.ndarray = np.asarray(y_train)

        forecast_mae: float = np.mean(np.abs(y_true_arr - y_pred_arr))

        if forecast_mae == 0:
            return 0.0

        scale: float = Metrics._calculate_naive_scale_mae(
            y_train_arr, seasonal_period=seasonal_period
        )

        if scale == 0:
            # If the scale is 0 (e.g., constant training data) and forecast_mae is not 0,
            # MASE is infinite. (forecast_mae == 0 case is handled above).
            return np.inf

        return forecast_mae / scale

    @staticmethod
    def rmsse(
        y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonal_period: int = 1
    ) -> float:
        """
        Calculates the Root Mean Squared Scaled Error (RMSSE).

        RMSSE is a scaled version of RMSE, similar to MASE but using squared errors.
        It's often used when larger errors are disproportionately penalized.

        Parameters
        ----------
        y_true : np.ndarray
            Array of true values for the forecast period.
        y_pred : np.ndarray
            Array of predicted values for the forecast period.
        y_train : np.ndarray
            Array of training data used to calculate the naive forecast scale.
        seasonal_period : int, default=1
            The seasonal period (m) for scaling. Use m=1 for non-seasonal data.

        Returns
        -------
        float
            The RMSSE value. Returns 0.0 if `forecast_rmse` is 0, and np.inf if the
            scaling factor is 0 but `forecast_rmse` is non-zero.
        """
        y_true_arr: np.ndarray = np.asarray(y_true)
        y_pred_arr: np.ndarray = np.asarray(y_pred)
        y_train_arr: np.ndarray = np.asarray(y_train)

        forecast_rmse: float = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))

        if forecast_rmse == 0:
            return 0.0

        scale: float = Metrics._calculate_naive_scale_rmse(
            y_train_arr, seasonal_period=seasonal_period
        )

        if scale == 0:
            return np.inf

        return forecast_rmse / scale

    @staticmethod
    def msis(
        y_true: np.ndarray,
        lower_interval: np.ndarray,
        upper_interval: np.ndarray,
        y_train: np.ndarray,
        quantile_level: float = 90.0,
        seasonal_period: int = 1,
    ) -> float:
        """
        Calculates the Mean Scaled Interval Score (MSIS) for probabilistic forecasts.

        The Interval Score for a single forecast quantifies the width of the prediction
        interval and penalizes observations falling outside the interval. MSIS scales
        this score by the MAE of a naive forecast.

        The Interval Score (IS) for a single point `Y` with a lower bound `L` and upper bound `U`
        at a confidence level `alpha` (where `alpha = 1 - quantile_level/100`) is:
        $IS = (U - L) + (2/alpha) * max(0, L - Y) + (2/alpha) * max(0, Y - U)$

        Parameters
        ----------
        y_true : np.ndarray
            Array of true values.
        lower_interval : np.ndarray
            Array of the lower bounds of the prediction intervals.
        upper_interval : np.ndarray
            Array of the upper bounds of the prediction intervals.
        y_train : np.ndarray
            Array of training data to calculate the naive forecast scale.
        quantile_level : float, default=90.0
            The confidence level as a percentage (e.g., 90 for a 90% prediction interval).
        seasonal_period : int, default=1
            The seasonal period (m) for scaling. Use m=1 for non-seasonal data.

        Returns
        -------
        float
            The average MSIS value. Returns 0.0 if `mean_interval_score` is 0, and
            np.inf if the scaling factor is 0 but `mean_interval_score` is non-zero.

        Raises
        ------
        ValueError
            If the `quantile_level` does not result in an alpha between 0 and 1 (exclusive).
        """
        y_true_arr: np.ndarray = np.asarray(y_true)
        lower_interval_arr: np.ndarray = np.asarray(lower_interval)
        upper_interval_arr: np.ndarray = np.asarray(upper_interval)
        y_train_arr: np.ndarray = np.asarray(y_train)

        alpha: float = 1.0 - (quantile_level / 100.0)
        if not (0 < alpha < 1):
            raise ValueError("Quantile level must result in alpha between 0 and 1 (exclusive).")

        interval_width: np.ndarray = upper_interval_arr - lower_interval_arr

        # Penalties for observations outside the interval
        penalty_lower: np.ndarray = np.where(y_true_arr < lower_interval_arr, lower_interval_arr - y_true_arr, 0)
        penalty_upper: np.ndarray = np.where(y_true_arr > upper_interval_arr, y_true_arr - upper_interval_arr, 0)

        interval_score_per_point: np.ndarray = interval_width + (2.0 / alpha) * (
            penalty_lower + penalty_upper
        )

        mean_interval_score: float = np.mean(interval_score_per_point)

        if mean_interval_score == 0:  # Very unlikely for real-world intervals
            return 0.0

        scale: float = Metrics._calculate_naive_scale_mae(
            y_train_arr, seasonal_period=seasonal_period
        )

        if scale == 0:
            return np.inf

        return mean_interval_score / scale

    @staticmethod
    def wql(
        y_true: np.ndarray,
        y_pred_median: np.ndarray,
        lower_quantile_pred: np.ndarray,
        upper_quantile_pred: np.ndarray,
        quantile_level: float = 90.0,
    ) -> float:
        """
        Calculates the Weighted Quantile Loss (WQL), also known as the average Pinball Loss
        across multiple quantiles. This implementation specifically considers the lower,
        median, and upper quantiles of a prediction interval.

        The quantiles are derived from the `quantile_level`:
        - Lower quantile: (100 - quantile_level) / 200.0
        - Median: 0.5
        - Upper quantile: (100 + quantile_level) / 200.0

        Parameters
        ----------
        y_true : np.ndarray
            Array of true values.
        y_pred_median : np.ndarray
            Array of predicted median values (50th percentile).
        lower_quantile_pred : np.ndarray
            Array of predicted lower quantile bounds.
        upper_quantile_pred : np.ndarray
            Array of predicted upper quantile bounds.
        quantile_level : float, default=90.0
            The level of the central prediction interval (e.g., 90 for a 90% interval).

        Returns
        -------
        float
            The average WQL value, which is the mean of the sum of Pinball Losses
            for the three quantiles, divided by 3 (to normalize for 3 quantiles).
        """
        y_true_arr: np.ndarray = np.asarray(y_true)
        y_pred_median_arr: np.ndarray = np.asarray(y_pred_median)
        lower_quantile_pred_arr: np.ndarray = np.asarray(lower_quantile_pred)
        upper_quantile_pred_arr: np.ndarray = np.asarray(upper_quantile_pred)

        tau_lower: float = (100.0 - quantile_level) / 200.0
        tau_median: float = 0.5
        tau_upper: float = (100.0 + quantile_level) / 200.0

        def pinball_loss(y: np.ndarray, y_hat: np.ndarray, tau: float) -> np.ndarray:
            """
            Calculates the Pinball Loss for a given quantile.

            Parameters
            ----------
            y : np.ndarray
                True values.
            y_hat : np.ndarray
                Predicted quantile values.
            tau : float
                The quantile level (e.g., 0.1 for 10th percentile).

            Returns
            -------
            np.ndarray
                Pinball loss for each data point.
            """
            error: np.ndarray = y - y_hat
            return np.where(error >= 0, tau * error, (tau - 1) * error)

        loss_lower: np.ndarray = pinball_loss(y_true_arr, lower_quantile_pred_arr, tau_lower)
        loss_median: np.ndarray = pinball_loss(y_true_arr, y_pred_median_arr, tau_median)
        loss_upper: np.ndarray = pinball_loss(y_true_arr, upper_quantile_pred_arr, tau_upper)

        # Sum of losses for each observation across the three quantiles, then average.
        total_loss_per_observation: np.ndarray = loss_lower + loss_median + loss_upper

        return np.mean(total_loss_per_observation) / 3.0

    @staticmethod
    def cfe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the Cumulative Forecasting Error (CFE).

        CFE measures the total bias of the forecast.
        A positive CFE indicates that demand was systematically underestimated.
        A negative CFE indicates that demand was systematically overestimated.

        Parameters
        ----------
        y_true : np.ndarray
            Array of true values.
        y_pred : np.ndarray
            Array of predicted values.

        Returns
        -------
        float
            The CFE value. Returns 0.0 if `y_true` is empty.
        """
        y_true_arr: np.ndarray = np.asarray(y_true)
        y_pred_arr: np.ndarray = np.asarray(y_pred)

        if y_true_arr.size == 0:
            return 0.0

        return float(np.sum(y_true_arr - y_pred_arr))

    @staticmethod
    def pis(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates Periods In Stock (PIS).

        PIS measures the accumulated sum of forecast errors over time, indicating
        how long errors persist. This is particularly useful for intermittent time series.
        PIS is defined as: $- \sum_{t=1}^{h} \sum_{j=1}^{t} (y_{true,j} - y_{pred,j})$
        or simply, the negative sum of cumulative errors at each time point.

        A positive PIS value indicates accumulated overstock (on average, negative cumulative errors).
        A negative PIS value indicates accumulated understock (on average, positive cumulative errors).
        Values closer to zero are better.

        Parameters
        ----------
        y_true : np.ndarray
            Array of true values.
        y_pred : np.ndarray
            Array of predicted values.

        Returns
        -------
        float
            The PIS value. Returns 0.0 if `y_true` is empty.
        """
        y_true_arr: np.ndarray = np.asarray(y_true)
        y_pred_arr: np.ndarray = np.asarray(y_pred)

        if y_true_arr.size == 0:
            return 0.0

        errors: np.ndarray = y_true_arr - y_pred_arr
        cumulative_errors: np.ndarray = np.cumsum(errors)

        pis_value: float = -np.sum(cumulative_errors)

        return float(pis_value)