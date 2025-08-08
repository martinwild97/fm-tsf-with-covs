# Chronos

| | |
| :--- | :--- |
| **Developer** | Amazon |
| **Paper** | [Chronos: Learning the language of time series](https://arxiv.org/abs/2403.07815) |
| **Code/Model** | [GitHub](https://github.com/amazon-science/chronos-forecasting), [Hugging Face](https://huggingface.co/collections/amazon/chronos-models-and-datasets-65f1791d630a8d57cb718444) |

---

## 🧠 Core Idea & Architecture

Chronos is a family of open sourced pre-trained models that adapts powerful concepts from Large Language Models (LLMs) to time series forecasting. It uniquely frames forecasting as a "next token prediction" task, similar to how an LLM predicts the next word in a sentence. The models were pre-trained on a large and diverse corpus, including 28 public datasets with ~890k time series (84B observations) and 1 million synthetic series generated via Gaussian processes.

There are two main architectural variants:

### 1. Original T5-based Chronos
This version treats a time series like a language using a process of tokenization:
* **Scaling:** Each time series is first normalized using **mean scaling**, where every value is divided by the mean of the absolute values of the entire series. This preserves the sign and relative magnitudes.
* **Quantization:** The scaled values are then grouped into a fixed number of discrete bins (e.g., 4096), with bin edges determined by the percentiles of the training data. Each bin corresponds to a "token" in the model's vocabulary.

A standard T5 Encoder-Decoder architecture is then trained using a cross-entropy loss function to predict the most likely sequence of future tokens.

### 2. Chronos-Bolt (Efficient Variant)
This is a newer, more efficient version that makes two key changes to the architecture:
* **Patch-Level Embedding:** Instead of processing individual time steps, Chronos-Bolt uses **patching**, dividing the time series into non-overlapping segments. This reduces the effective sequence length, leading to significant gains in speed (up to 250x faster) and memory efficiency (up to 20x better).
* **Direct Multi-Step Forecasting:** It replaces the original's autoregressive, one-step-ahead token prediction with direct multi-step forecasting using **quantile regression**. This accelerates inference and directly provides probabilistic forecasts.

---

## ✨ Special Features & Data Handling

* **Probabilistic Forecasting:** Chronos is a leader in probabilistic forecasting. The original version predicts full probability distributions over its vocabulary of tokens, while Chronos-Bolt directly outputs various quantiles for multi-step probabilistic forecasts.

* **Modular Covariate Support:** At its core, Chronos is a **univariate** model. The integration of exogenous variables is handled through a modular, **external/residual modeling** approach, especially within frameworks like **AutoGluon**. The process works as follows:
    1. A separate tabular model (e.g., LightGBM) is trained to predict the target series using only the known exogenous variables.
    2. The predictions from this regressor are subtracted from the actual values to compute the **residuals**.
    3. The Chronos model is then used to produce a univariate forecast of these residuals.
    4. The final forecast is the sum of the regressor's prediction and Chronos's residual forecast.

---

## 👍 Strengths

* **State-of-the-Art Probabilistic Forecasts:** Chronos is widely considered one of the best open models for tasks requiring uncertainty quantification.
* **Ease of Use:** When used via the **AutoGluon** package, Chronos is very user-friendly. It requires minimal configuration and handles data preparation like imputation, model selection, and even fine-tuning automatically.
* **CPU Optimization (Bolt Variant):** The Chronos-Bolt variant is highly optimized for performance and can run efficiently on CPUs, making it accessible without specialized hardware.
* **Effective Covariate Integration:** The external regressor approach proved to be a reliable and effective strategy for integrating exogenous variables in extensive benchmarks.

---

## 👎 Weaknesses & Practical Tips

* **Computationally Intensive (Original T5 Models):** The original token-based models can be slow and memory-intensive due to their autoregressive, step-by-step generation process. For practical use, the **Bolt variant is highly recommended**.
* **Indirect Covariate Handling:** The modular approach for covariates, while effective, is not a deep integration. This might prevent the model from capturing complex, non-additive interactions between the target series and external factors.
* **Requires Future-Known Covariates:** The external regressor approach requires that the values for all dynamic exogenous variables are known for the entire forecast horizon, which is a significant limitation in many real-world scenarios. The suggested "hacky" workarounds (like lagging or pre-forecasting covariates) can introduce additional errors.
* **Information Loss via Quantization:** The tokenization process in the original models relies on a fixed vocabulary and can lead to a loss of precision for time series with very wide dynamic ranges.
* **Sensitivity to Configuration:** While patching in Chronos-Bolt improves efficiency, the optimal patch size can be data-dependent and may require careful consideration. Similarly, the effectiveness of mean scaling can vary for series with strong trends or non-stationarities.

---

## 💻 Code Example

This file contains the wrapper class used to serve the Chronos model within the **FastAPI service** for the thesis benchmark. It demonstrates a practical implementation using the `autogluon.timeseries` library, including the complete logic for handling the **external regressor** to integrate both past-known and future-known covariates.


➡️ **[View the Chronos implementation file](../../code/fm-api/src/models/chronos.py)**

---

## 📏 Available Model Sizes

Chronos is available in a range of sizes for both the original T5 and the newer Bolt architectures:

| Model Name | Parameters | Based On |
| :--- | :--- | :--- |
| **chronos-t5-tiny** | 8M | t5-efficient-tiny |
| **chronos-t5-mini** | 20M | t5-efficient-mini |
| **chronos-t5-small** | 46M | t5-efficient-small |
| **chronos-t5-base** | 200M | t5-efficient-base |
| **chronos-t5-large** | 710M | t5-efficient-large |
| **chronos-bolt-tiny** | 9M | t5-efficient-tiny |
| **chronos-bolt-mini** | 21M | t5-efficient-mini |
| **chronos-bolt-small** | 48M | t5-efficient-small |
| **chronos-bolt-base** | 205M | t5-efficient-base |