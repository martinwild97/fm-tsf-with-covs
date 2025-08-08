from typing import Optional, Dict, Any, List
import re
import time
import pandas as pd

import src.api.schemas

# Define constants for frequency processing.
# Typical seasonality representing periods within the next larger common unit.
SEASONALITY_MAP_DATA: Dict[str, Optional[int]] = {
    "H": 24,  # Hours per Day
    "D": 7,  # Days per Week
    "W": 52,  # Weeks per Year (approx)
    "M": 12,  # Months per Year
    "Q": 4,  # Quarters per Year
    "Y": 1,  # Years per Year (seasonality 1 means no clear intra-year pattern)
    "A": 1,  # Alias for Year
}

# Tokens used for TimesFM frequency grouping.
# More information: https://github.com/google-research/timesfm/tree/master?tab=readme-ov-file#perform-inference
GROUP2_TOKENS: List[str] = ["Q", "Y", "A"]  # Quarterly and yearly
GROUP1_TOKENS: List[str] = ["W", "M"]  # Weekly and monthly (excluding Minute)
GROUP0_TOKENS: List[str] = [
    "H",
    "T",
    "MIN",
    "S",
    "L",
    "U",
    "N",
    "D",
]  # Sub-daily and daily


# Set of known base frequency categories after normalization.
KNOWN_BASE_CATEGORIES: set[str] = {
    "H",
    "T",
    "MIN",
    "S",
    "L",
    "U",
    "N",
    "D",
    "W",
    "M",
    "Q",
    "Y",
    "A",
}


class FoundationModel:
    """Base class for foundation models.

    Provides common functionality for model initialization, loading,
    forecasting, time measurement, and frequency processing. Subclasses
    must implement the `load_model` and `forecast` methods.

    Parameters
    ----------
    model_path : Optional[str], default=None
        Path to the pre-trained model or the name of a pre-trained model
        from a model repository.
    **kwargs : Any
        Additional keyword arguments to be stored and potentially used by
        subclasses during model loading or forecasting.
    """

    def __init__(self, model_path: Optional[str] = None, **kwargs: Any) -> None:
        """Initializes the FoundationModel base class.

        Stores the model path and keyword arguments and attempts to load
        the model using the `load_model` method, which must be implemented
        by subclasses.

        Parameters
        ----------
        model_path : Optional[str], default=None
            Path to the pre-trained model or the name of a pre-trained model.
        **kwargs : Any
            Additional keyword arguments to be stored.
        """
        self.model_path: Optional[str] = model_path
        self.kwargs: Dict[str, Any] = kwargs
        self.model: Optional[Any] = None  # Attribute to hold the loaded model instance.
        self.start_time_seconds: float = (
            0.0  # Time in seconds when measurement started.
        )

        # Attempt to load the model upon initialization.
        self.load_model(self.model_path, **self.kwargs)

    def load_model(self, model_path: Optional[str] = None, **kwargs: Any) -> None:
        """Loads a pre-trained model.

        This method must be implemented by subclasses. It should handle the
        logic for loading a specific model based on the `model_path` and
        optional keyword arguments.

        Parameters
        ----------
        model_path : Optional[str], default=None
            Path to the pre-trained model or name of a pre-trained model
            from the model repository.
        **kwargs : Any
            Additional parameters for model loading.

        Raises
        ------
        NotImplementedError
            This method is abstract and must be implemented by a subclass.
        """
        ...  # Indicate intentional empty implementation for abstract method.

    def forecast(
        self, input_data: src.api.schemas.ModelInput
    ) -> src.api.schemas.Forecast:
        """Creates a forecast for given actuals.

        This method must be implemented by subclasses. It should take a
        `ModelInput` object and return a `Forecast` object containing
        prediction results.

        Parameters
        ----------
        input_data : src.api.schemas.ModelInput
            ModelInput object containing the time series data and forecast
            parameters.

        Returns
        -------
        src.api.schemas.Forecast
            Forecast object with median, lower (input.quantile), and upper
            (1 - input.quantile) quantiles.

        Raises
        ------
        NotImplementedError
            This method is abstract and must be implemented by a subclass.
        """
        ...  # Indicate intentional empty implementation for abstract method.

    def _start_time_measurement(self) -> None:
        """Starts the time measurement for a process.

        Records the current time in seconds.
        """
        self.start_time_seconds = time.time()

    def _end_time_measurement(self) -> float:
        """Ends the time measurement and returns the elapsed time in seconds.

        Returns
        -------
        float
            Elapsed time in seconds since the last call to
            `_start_time_measurement`.
        """
        return time.time() - self.start_time_seconds

    def _process_frequency(self, freq_str: str) -> Dict[str, Any]:
        """Processes a Pandas frequency string to extract base frequency and related information.

        Extracts the base frequency, typical seasonality, and frequency group
        (relevant for models like TimesFM). Handles case, multiples, anchoring
        suffixes, business days ('B', 'C'), and start/end markers ('S'/'E').

        Parameters
        ----------
        freq_str : str
            The Pandas frequency string (e.g., 'H', 'd', 'W-MON', 'MS', '2Q',
            'T', 'B').

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
            - 'base': The processed base frequency string (e.g., 'H', 'D', 'M', 'T').
            - 'seasonality': An integer representing a typical seasonality, or None
              if not applicable or defined for the base frequency.
            - 'group': An integer representing a frequency group (e.g., 0, 1, 2),
              or None if not applicable or defined for the base frequency.

        Raises
        ------
        ValueError
            If the input is not a valid string or if the base frequency
            cannot be determined or is fundamentally unsupported after
            normalization.
        """
        if not isinstance(freq_str, str) or not freq_str:
            raise ValueError("Frequency must be a non-empty string.")

        # --- Normalize the frequency string ---
        processed_freq = freq_str.upper()
        # Keep the original string for potential error messages.
        original_freq_for_error = freq_str

        # 1. Remove anchoring suffixes (e.g., -MON, -DEC).
        processed_freq = processed_freq.split("-")[0]

        # 2. Remove leading digits (multiples like '2H', '7D') - keep base unit.
        processed_freq = re.sub(r"^\d+", "", processed_freq)

        # 3. Handle Business day ('B', 'C') -> Map to Day ('D') for consistency in seasonality/grouping.
        # Standalone B, C, CB map to D. Prefixes like BQ will be handled by S/E logic later.
        if processed_freq in ("B", "C", "CB"):
            # Treat standalone business/custom day like Calendar Day for seasonality/grouping.
            processed_freq = "D"
        # Note: If processed_freq starts with 'B' or 'CB' and is longer than 1 char (e.g., BQS),
        # the prefix removal and S/E handling in step 4 will process it correctly (e.g., BQS -> QS -> Q).

        # 4. Handle Start/End markers ('S'/'E') -> Map to base period (M, Q, Y/A).
        if len(processed_freq) > 1 and processed_freq.endswith(("S", "E")):
            base = processed_freq[:-1]
            # Map Month, Quarter, Year start/end markers to their base.
            # Use 'A' as an alias for 'Y' (Year).
            if base in ["M", "Q", "Y", "A"]:
                processed_freq = base
            # Handle semi-month start/end ('SM') -> map to Month ('M').
            elif base == "SM":
                processed_freq = "M"
            # Handle Business versions like BQS -> QS after B mapping in step 3.
            elif base.startswith(("B", "CB")):
                base_no_b = re.sub(r"^(C?B)", "", base)
                if base_no_b in ["M", "Q", "Y", "A"]:
                    processed_freq = base_no_b
            # Note: W, D, H, etc. ending in S/E are generally not mapped to their base here.

        # --- Determine seasonality ---
        # Look up the typical seasonality from the predefined map.
        seasonality: Optional[int] = SEASONALITY_MAP_DATA.get(processed_freq)

        # --- Determine Group (based on TimesFM logic) ---
        group: Optional[int] = None
        # Group 2: Quarterly and yearly frequencies (Q, Y, A).
        # BQE/BQS etc. should have been normalized to Q/Y/A by step 4.
        if any(processed_freq.startswith(token) for token in GROUP2_TOKENS):
            group = 2
        else:
            # Group 1: Weekly and monthly frequencies (W, M).
            # Need careful check for Minute ('MIN') vs Month ('M').
            if any(
                processed_freq.startswith(token) for token in GROUP1_TOKENS
            ) and not processed_freq.startswith("MIN"):  # Exclude Minute
                group = 1
            else:
                # Group 0: Sub-daily and daily frequencies.
                # Includes H, T(min), MIN, S, L(ms), U(us), N(ns), D(day).
                # B/C were mapped to D already.
                if any(processed_freq.startswith(token) for token in GROUP0_TOKENS):
                    group = 0

        # --- Final Check and Return ---
        # Raise error only if the frequency remains completely uninterpretable after normalization.
        # Check if the processed frequency is in the set of known base categories.
        if processed_freq not in KNOWN_BASE_CATEGORIES:
            raise ValueError(
                f"Could not determine seasonality or group for frequency '{original_freq_for_error}'. "
                f"Resulting base frequency '{processed_freq}' is unsupported or "
                "could not be classified after normalization."
            )

        # Return the extracted information.
        return {"base": processed_freq, "seasonality": seasonality, "group": group}

    def _prep_dict_with_df_json(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Prepares a dictionary for JSON serialization by converting DataFrames to lists.

        Recursively traverses the dictionary and converts any pandas DataFrames
        found to dictionaries with orientation 'list', which is suitable for
        JSON serialization.

        Parameters
        ----------
        input_dict : Dict[str, Any]
            The input dictionary to be prepared, potentially containing pandas
            DataFrames.

        Returns
        -------
        Dict[str, Any]
            The prepared dictionary with DataFrames converted to dictionaries
            where keys are column names and values are lists of data.
        """
        output_dict: Dict[str, Any] = {}
        for key, value in input_dict.items():
            if isinstance(value, pd.DataFrame):
                # Convert DataFrame to a dictionary of lists.
                output_dict[key] = value.to_dict(orient="list")
            elif isinstance(value, dict):
                # Recursively process nested dictionaries.
                output_dict[key] = self._prep_dict_with_df_json(value)
            else:
                # Keep other types as they are.
                output_dict[key] = value
        return output_dict
