import re
from typing import Dict, Set

# --- Constants for Frequency Processing ---

# Defines standard base frequency aliases after normalization,
# using non-deprecated Pandas terminology (e.g., 'h' for hour instead of 'H').
KNOWN_BASE_PERIOD_ALIASES: Set[str] = {
    'B', 'D', 'W', 'M', 'Q', 'Y',  # Business day, Day, Week, Month, Quarter, Year
    'h', 'min', 's', 'ms', 'us', 'ns' # Hour, Minute, Second, Millisecond, Microsecond, Nanosecond
}

# Maps deprecated or less common Pandas frequency aliases to their
# normalized, non-deprecated, and commonly accepted counterparts.
DEPRECATION_MAP: Dict[str, str] = {
    'H': 'h',    # Deprecated Hour to current 'h'
    'T': 'min',  # Deprecated Minute to current 'min'
    'S': 's',    # Deprecated Second to current 's'
    'L': 'ms',   # Deprecated Millisecond to current 'ms'
    'U': 'us',   # Deprecated Microsecond to current 'us'
    'N': 'ns',   # Deprecated Nanosecond to current 'ns'
    'A': 'Y',    # Deprecated Annual to current 'Y' (Year)
    'BH': 'h',   # Business Hour (deprecated) to Hour
    'CBH': 'h',  # Custom Business Hour (deprecated) to Hour
}

# Maps common Pandas offset aliases (start/end of period, business versions)
# to their corresponding base frequency alias. Order matters for specificity.
OFFSET_ALIAS_MAP: Dict[str, str] = {
    # Semi-Month (SM) aliases map to Month ('M')
    "SMS": "M", "SME": "M",
    # Month Start/End (MS, ME) and their Business (BMS, BME) and Custom Business (CBMS, CBME) versions map to Month ('M')
    "MS": "M", "ME": "M", "BMS": "M", "BME": "M", "CBMS": "M", "CBME": "M",
    # Quarter Start/End (QS, QE) and their Business (BQS, BQE) versions map to Quarter ('Q')
    "QS": "Q", "QE": "Q", "BQS": "Q", "BQE": "Q",
    # Year Start/End (YS, YE) and their Business (BYS, BYE) versions map to Year ('Y')
    "YS": "Y", "YE": "Y", "BYS": "Y", "BYE": "Y",
    # Deprecated Annual Start/End (AS, AE) also map to Year ('Y')
    "AS": "Y", "AE": "Y",
}


def extract_base_frequency(freq_str: str) -> str:
    """
    Processes a Pandas frequency string to extract its normalized, non-deprecated base period alias.

    This function handles a wide range of Pandas frequency string variations, including:
    - Case insensitivity (by converting to uppercase initially).
    - Multiples (e.g., '2H', '7D', '3W').
    - Anchoring suffixes (e.g., 'W-MON', 'Q-DEC').
    - Offset aliases (e.g., 'ME' for month end, 'QS' for quarter start, 'BMS' for business month start).
    - Business day variations (e.g., 'B' for business day, 'C'/'CB' for custom business day).
    - Deprecated Pandas aliases (e.g., 'H' for hour, 'T' for minute, 'A' for annual).

    The function aims to return a consistent, normalized base frequency alias that is part of
    `KNOWN_BASE_PERIOD_ALIASES` (e.g., 'h', 'D', 'W', 'M', 'Q', 'Y', 'min', 's').

    Parameters
    ----------
    freq_str : str
        The Pandas frequency string to process (e.g., 'H', '2D', 'W-SUN', 'M', 'Q-MAR', '5min', 'B', 'CB').

    Returns
    -------
    str
        The normalized base period alias string. For time units (hour, minute, second, etc.),
        it returns lowercase (e.g., 'h', 'min'). For calendar/business units, it returns uppercase
        (e.g., 'D', 'W', 'M', 'Q', 'Y', 'B').

    Raises
    ------
    ValueError
        If the input `freq_str` is not a valid non-empty string, or if the base period alias
        cannot be determined from the input, or if the resulting normalized alias is
        fundamentally unsupported.
    """
    if not isinstance(freq_str, str) or not freq_str:
        raise ValueError("Input frequency must be a non-empty string.")

    # Store the original frequency string for informative error messages.
    original_freq_for_error: str = freq_str

    # --- Step 1: Normalize the frequency string for processing ---
    processed_freq: str = freq_str.upper()

    # Step 2: Remove anchoring suffixes (e.g., '-MON', '-DEC')
    processed_freq = processed_freq.split('-')[0]

    # Step 3: Remove leading digits (multiples like '2H', '7D', '3W')
    processed_freq = re.sub(r'^\d+', '', processed_freq)

    # Step 4: Handle specific offset aliases (e.g., 'MS', 'QE', 'BMS')
    processed_freq = OFFSET_ALIAS_MAP.get(processed_freq, processed_freq)

    # Step 5: Handle other specific frequency string patterns.
    if processed_freq in ('C', 'CB'):
        processed_freq = 'B'
    # Deprecated Business Hours (BH, CBH) map to 'h' (hour).
    elif processed_freq == 'BH' or processed_freq == 'CBH':
        processed_freq = 'h'

    # Step 6: Apply general deprecation mapping for remaining common aliases.
    processed_freq = DEPRECATION_MAP.get(processed_freq, processed_freq)

    # Step 7: Ensure consistency in casing for final base aliases.
    if processed_freq.lower() in ['h', 'min', 's', 'ms', 'us', 'ns']:
        processed_freq = processed_freq.lower()

    # --- Final Validation ---
    if processed_freq not in KNOWN_BASE_PERIOD_ALIASES:
        raise ValueError(
            f"Could not determine a supported base period alias for the input frequency "
            f"'{original_freq_for_error}'. The normalized result '{processed_freq}' is either "
            "unsupported or could not be reliably classified. Please check the input string."
        )

    return processed_freq