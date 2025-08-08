import os
import sys
import multiprocessing
from pathlib import Path
from typing import List, Tuple, Dict, Any

current_dir: str = os.path.dirname(os.path.abspath(__file__))
project_root: str = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Project-Specific Imports ---
from src.forecast import Forecaster, MODEL_SETUPS, FORECAST_DIR
from src.data.data_loader import DataLoader # Assuming DataLoader is needed by Forecaster.

# --- Global Configuration Constants ---
# File paths for storing forecast metadata and detailed data.
CPU_META_PATH: Path = FORECAST_DIR / "forecasts_meta_cpu.csv"
CPU_DATA_PATH: Path = FORECAST_DIR / "forecasts_data_cpu.csv"
API_GPU_META_PATH: Path = FORECAST_DIR / "forecasts_meta_api_gpu.csv"
API_GPU_DATA_PATH: Path = FORECAST_DIR / "forecasts_data_api_gpu.csv"

# Identifiers for different processing types, used in logs and file names.
PROCESS_TYPE_CPU: str = "cpu"
PROCESS_TYPE_GPU: str = "gpu"

# Model categories that run on CPU.
CPU_MODEL_CATEGORIES: List[str] = ["baseline", "other"]
# Model categories that use API/GPU.
API_GPU_MODEL_CATEGORIES: List[str] = ["fm_api", "timegpt"]


# --- Dynamic Model Key Filtering ---
# Lists to store keys of models belonging to CPU or API/GPU categories.
cpu_model_keys: List[str] = []
api_gpu_model_keys: List[str] = []

for key, setup in MODEL_SETUPS.items():
    if setup.model_category in CPU_MODEL_CATEGORIES:
        cpu_model_keys.append(key)
    elif setup.model_category in API_GPU_MODEL_CATEGORIES:
        api_gpu_model_keys.append(key)

# Specific GPU model keys to potentially group or exclude in configurations.
MAIN_GPU_KEYS: List[str] = [
    "ttm",
    "TiDE",
    "NBEATSx",
    "moirai_base",
    "timesfm_v2",
]
OTHER_GPU_KEYS: List[str] = [
    "moirai_moe",
    "TFT",
    "ttm_finetuned"
]

# --- Multiprocessing Helper Function ---
def run_forecaster(forecaster_instance: Forecaster, config_list: List[Tuple[List[str], List[str]]]) -> None:
    """
    Target function for a multiprocessing `Process`. It runs the `forecast_all`
    method of a `Forecaster` instance with a given configuration.

    Parameters
    ----------
    forecaster_instance : Forecaster
        An initialized `Forecaster` object. This object should be configured
        for a specific process (e.g., CPU or GPU) and its output paths.
    config_list : List[Tuple[List[str], List[str]]]
        A list of configuration tuples, where each tuple specifies a list of
        model keys and a list of covariate modes to run.
    """
    # Using print here for immediate visibility in console for multiprocessing.
    print(f"Process {os.getpid()} starting with config: {config_list}")
    forecaster_instance.forecast_all(config=config_list)
    print(f"Process {os.getpid()} finished.")


# --- Main Execution Block ---
if __name__ == '__main__':
    """
    Main entry point for initiating time series forecasting runs.

    This block sets up `Forecaster` instances for different processing types (CPU, API/GPU),
    defines the forecasting configurations (models and covariate modes), and
    executes the forecasts. It includes commented-out examples for running
    forecasts in parallel using `multiprocessing`.
    """

    # --- Initialize Forecaster Instances ---
    data_loader_instance = DataLoader()

    # Initialize Forecaster for CPU-bound models.
    forecaster_cpu = Forecaster(
        data_loader=data_loader_instance,
        meta_path=CPU_META_PATH,
        data_path=CPU_DATA_PATH,
        process=PROCESS_TYPE_CPU,
        redo_failed_forecasts=True, # Set to True to re-attempt previously failed CPU forecasts.
    )

    # Initialize Forecaster for API/GPU-bound models.
    forecaster_api_gpu = Forecaster(
        data_loader=data_loader_instance,
        meta_path=API_GPU_META_PATH,
        data_path=API_GPU_DATA_PATH,
        process=PROCESS_TYPE_GPU,
        redo_failed_forecasts=False,
    )

    # --- Define Forecasting Configurations ---
    config_cpu: List[Tuple[List[str], List[str]]] = [
        ([k for k in cpu_model_keys if k != "arima"], ["no"]),
        ([k for k in cpu_model_keys if k != "all_baseline"], ["all"]),
        (["chronos_base", "chronos_gbm"], [
            "noise",
            "only_past",
            "only_future",
            "time",
            "lagged_target",
        ]),
    ]

    config_api_gpu: List[Tuple[List[str], List[str]]] = [
        (MAIN_GPU_KEYS, ["no"]),
        (MAIN_GPU_KEYS, ["all"]),
        ([k for k in MAIN_GPU_KEYS if k not in ["TiDE", "NBEATSx", "TFT"]], [
            "noise",
            "only_past",
            "only_future",
            "time",
            "lagged_target",
        ]),
        (OTHER_GPU_KEYS, ["no"]),
        (OTHER_GPU_KEYS, ["all"]),
        ([k for k in OTHER_GPU_KEYS if k not in ["TiDE", "NBEATSx", "TFT"]], [
            "noise",
            "only_past",
            "only_future",
            "time",
            "lagged_target",
        ]),
    ]

    # --- Execute Forecasts ---
    forecaster_api_gpu.forecast_all(config=config_api_gpu)

