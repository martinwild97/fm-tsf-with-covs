# Foundation Model Time Series API

## Getting Started

The application can be started by calling

```
docker compose up
```
Initial startup might take a while, as the required model weights have to be downloaded. The application should print something like `Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)` when it's running and ready to receive forecasting requests. An example request to the application could look like:

### Base Example
```
curl -X POST "http://localhost:8080/" -H "Content-Type: application/json" -d '{"forecast_horizon": 3, "actuals":{"frequency": "M", "timestamp": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"], "data":{ "value": [1, 2, 3, 4, 5]}, "timestamp_format": "%Y-%m-%d"}}'
```
### Multi TS Example
```
curl -X POST "http://localhost:8080/" -H "Content-Type: application/json" -d '{"forecast_horizon": 3, "actuals":{"frequency": "M", "timestamp": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"], "data":{ "value1": [1, 2, 3, 4, 5], "value2": [1, 2, 3, 4, 5]}}}'
```
### Covariate Example
```
curl -X POST "http://localhost:8080/" -H "Content-Type: application/json" -d '{"forecast_horizon": 3, "actuals":{"frequency": "M", "timestamp": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01"], "data":{ "value": [1, 2, 3, 4, 5], "future_cov": [1, 2, 3, 4, 5, 6, 7, 8], "past_cov": [1, 2, 3, 4, 5]}, "future_cov_names": ["future_cov"], "past_cov_names": ["past_cov"]}}'
```

Length of future covariate has to be equal to timestamp length. Length of values and past covariate has to be length of timestamps - forecast horizon.

### Multivariate Example
```
curl -X POST "http://localhost:8080/" -H "Content-Type: application/json" -d '{"forecast_horizon": 3, "actuals":{"frequency": "M", "timestamp": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"], "data":{ "value1": [1, 2, 3, 4, 5], "value2": [1, 2, 3, 4, 5]}, "multivariate": 1}}'
```

>Use pandas [period aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases) to pass the frequency. Passing the frequency is optional but recommended.

>You can also pass the `timestamp_format` e.g.: ..., "timestamp_format": "%d/%m/%Y", ... (also optional but recommended)

>TimesFM only supports univariate forecasts. The other models also support multivariate forecasts.

>DL-Models implemented via Nixtlas [neuralforecast](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html) package. The models use default parameters.

## Available Models

| Model       | Path                                          |
|------------|----------------------------------------------|
| TinyTimeMixer | ibm-granite/granite-timeseries-ttm-r2    |
|            | ibm-granite/granite-timeseries-ttm-r1       |
| TimesFM    | google/timesfm-1.0-200m-pytorch             |
|            | google/timesfm-2.0-500m-pytorch             |
| Moirai     | Salesforce/moirai-moe-1.0-R-small           |
|            | Salesforce/moirai-moe-1.0-R-base            |
|            | Salesforce/moirai-1.0-R-small               |
|            | Salesforce/moirai-1.1-R-small               |
|            | Salesforce/moirai-1.0-R-base                |
|            | Salesforce/moirai-1.1-R-base                |
|            | Salesforce/moirai-1.0-R-large               |
|            | Salesforce/moirai-1.1-R-large               |
| DL Models  | NHITS                                       |
|            | TFT                                         |
|            | TiDE                                        |
|            | NBEATSx                                     |

>Hint: Check huggingface for new model versions.

* Get currently deployed model version:
```
curl -X GET "http://localhost:8080/model"
```
* Switch model
```
curl -X POST "http://localhost:8080/model" -H "Content-Type: application/json" -d '{"model_type": "ttm", "model_path": "ibm-granite/granite-timeseries-ttm-r2"}'
```
> The `switch_model` also takes additional model-specific parameters. E.g.: for Moirai: "patch_size": 64

> TTM provides besides zero-shot forecasts few-shot fine-tuning: set `fine_tune_perc`. E.g.: "fine_tune_perc": 0.5 will use 50% of the provided dataset for finetuning.

## Development

The project comes with a devcontainer for VSCode. Running the container will take care to already install required dependencies in the environment, in order to start directly with coding. 

In order to add new requirements, or alter existing requirements, use the [requirements.in](requirements.in) file. Afterwards, in order to update also the fixed subdependencies of the [requirements.txt](requirements.txt) file, execute `pip-compile`. The new dependency set can then be installed by `pip install -r requirements.txt`.

The fastAPI server can also be started in the devcontainer for development or debugging via
```
uvicorn src.api.app:app --host 0.0.0.0 --port 8080 --reload
```
