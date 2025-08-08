# TimesFM

| | |
| :--- | :--- |
| **Developer** | Google |
| **Paper** | [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688) |
| **Code/Model** | [GitHub](https://github.com/google-research/timesfm), [HuggingFace](https://huggingface.co/google/timesfm-2.0-500m-pytorch) |

---

## 🧠 Core Idea & Architecture

TimesFM (Time Series Foundation Model) is a powerful, open-source model designed for long-horizon time series forecasting. It was pre-trained on a massive corpus of approximately 100 billion data points, including real-world sources like Google Trends and Wikipedia Pageviews, which allows it to learn a wide array of universal temporal patterns.

The core architecture is a **decoder-only Transformer**, similar to GPT models in NLP, which is highly effective for sequence generation tasks. A key characteristic of TimesFM is its use of **patching**.

The model consists of stacked Transformer layers with causal masking and residual connections, ensuring that predictions for a given time step only depend on past information.

---

## ✨ Special Features & Data Handling

* **Point Forecasting Specialist:** TimesFM is highly optimized for **point forecasting** and is considered one of the best models for this task. While the 2.0 version offers experimental, uncalibrated quantile heads, it is not primarily designed for reliable probabilistic forecasting.

* **Modular Covariate Support:** TimesFM is fundamentally a **univariate** model. It integrates exogenous variables through a distinct, modular strategy using an external linear regressor library (`xreg_lib`), which notably **requires the JAX framework**. Two primary methods are offered:
    1.  **`timesfm + xreg`:** The TimesFM model first generates a univariate forecast. A linear model is then trained to predict the **residuals** (the errors of the initial forecast) using the exogenous variables. The final forecast is the sum of the initial forecast and the predicted residuals.
    2.  **`xreg + timesfm`:** A linear model first predicts the target series using the exogenous variables. TimesFM is then used to forecast the **residuals** of this initial linear model. The final forecast is the sum of the linear model's prediction and the residual forecast from TimesFM.

---

## 👍 Strengths

* **State-of-the-Art Point Forecasts**
* **Large-Scale Pre-training:** Training on 100 billion data points from diverse sources gives it strong generalization capabilities.
* **Efficient Architecture:** The decoder-only and patching design makes it very fast and capable of handling long historical contexts (up to 2048 steps).

---

## 👎 Weaknesses & Practical Tips

* **Limited Probabilistic Forecasting:** The model's primary weakness is its lack of robust, calibrated probabilistic outputs.
* **Requires Future-Known Covariates:** The `xreg` approach requires that the values for all dynamic exogenous variables are known for the entire forecast horizon, which is a significant limitation in many real-world scenarios. The suggested "hacky" workarounds (like lagging or pre-forecasting covariates) can introduce additional errors.
* **JAX Dependency for Covariates:** While the core model is available in PyTorch, using the exogenous variable functionality requires JAX, which can be an extra dependency for some users.
* **No Native Handling of Missing Values:** The model cannot process time series with missing (NaN) values. Data must be complete, requiring a separate imputation or filling step during preprocessing before being fed to the model.
* **Complex Finetuning:** Fine-tuning is supported for both JAX and PyTorch, but users have reported challenges with the process.

---

## 💻 Code Example

This file contains the wrapper class for serving the TimesFM model in the thesis project's FastAPI service. It uses the official `timesfm` library and demonstrates the model's modular approach to handling external variables via the `forecast_with_covariates` function. The implementation includes the logic to pre-forecast any past-known covariates to make them available for the future horizon.

➡️ **[View the TimesFM implementation file](../../code/fm-api/src/models/timesfm.py)**

---

## 📏 Available Model Sizes

TimesFM is available in two main versions, both supporting PyTorch and JAX.

| Model Name | Parameters | Max Context Length |
| :--- | :--- | :--- |
| **TimesFM-1.0** | 200M | 512 |
| **TimesFM-2.0** | 500M | 2048 |