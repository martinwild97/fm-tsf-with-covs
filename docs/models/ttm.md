# TTM (Tiny Time Mixers)

| | |
| :--- | :--- |
| **Developer** | IBM |
| **Paper** | [Tiny Time Mixers (TTMs): Fast pre-trained models for enhanced zero/few-shot forecasting of multivariate time series](https://arxiv.org/abs/2401.03955) |
| **Code/Model** | [GitHub](https://github.com/ibm-granite/granite-tsfm/tree/main/tsfm_public/models/tinytimemixer), [Hugging Face](https://huggingface.co/collections/ibm-granite/granite-time-series-models-663a90c6a2da73482bce3dc6) |

---

## 🧠 Core Idea & Architecture

Tiny Time Mixers (TTM) are a family of compact and efficient pre-trained models, representing the first "tiny" foundation models for TSF with sizes starting from just 1M parameters. The architecture is based on the lightweight **TSMixer**, which uses simple Multi-Layer Perceptron (MLP) blocks instead of complex self-attention mechanisms, making it a powerful non-Transformer alternative.

The core TTM model has a multi-level design:
* **TTM Backbone:** The primary learning module, built from TSMixer blocks. It uses innovative techniques to handle data with varying resolutions.
* **TTM Decoder:** A much smaller version of the backbone that can be adapted during fine-tuning to handle multivariate correlations.
* **Forecast Head:** A final linear layer that produces the point forecast.
* **Exogenous Mixer:** An optional, dedicated module for integrating external variables during fine-tuning.

A key design philosophy is the focus on **pre-training with resolution diversity** over sheer data volume, allowing these compact models to achieve strong generalization and state-of-the-art zero-shot performance.

---

## ✨ Special Features & Data Handling

* **Non-Transformer Architecture:** By using an all-MLP design, TTM is significantly smaller and faster than most Transformer-based FMs, making it suitable for resource-constrained environments, including laptops and CPU-only inference.

* **Techniques for Data Diversity:** TTM introduces several innovations to learn effectively from heterogeneous pre-training data:
    * **Adaptive Patching (AP):** Different layers of the backbone operate with varying patch lengths, helping the model generalize across time series with different granularities.
    * **Diverse Resolution Sampling (DRS):** A data augmentation technique that creates lower-resolution versions of existing series to ensure the model is exposed to a wide range of sampling frequencies.

* **Modular Covariate Support:** TTM supports all types of covariates (past and future-known) via a dedicated **Exogenous Mixer Block**. This module is activated during the fine-tuning stage to learn the influence of external variables and produce a refined forecast.

---

## 👍 Strengths

* **Compactness and Efficiency:** With models starting at just 1M parameters, TTMs are extremely lightweight, enabling fast pre-training and inference even on a single GPU or CPU.
* **Dedicated Exogenous Variable Integration:** The Exogenous Mixer Block provides an explicit, deep-learning-based mechanism to incorporate covariates during fine-tuning.

---

## 👎 Weaknesses & Practical Tips

* **Only supports finer frequencys:** TTM is trained mainly on minutely and hourly sampled data. The R2 version adds support for daily and weekly time series. Other frequencys can be forcastet but give poor results.
* **Point Forecasts Only:** TTM is designed to produce a single point forecast and does not natively support probabilistic forecasting to quantify uncertainty.
* **Fine-Tuning Required for Covariates:** The Exogenous Mixer is only activated during the fine-tuning stage, meaning the model cannot leverage covariates in a pure zero-shot setting. This fine-tuning process can also be time-consuming.
* **Sensitivity to Data Patterns:** The model can struggle with time series that have long sequences of zero values, which can cause its instance normalization to become unstable and lead to forecast failures.
* **Complex Model Selection:** TTM is available in multiple releases (R1, R2, R2.1) with many pre-trained configurations for different context and prediction lengths. While there is a function for model selection provided, the optimal choice can be data-dependent and may require experimentation.

---

## 🚀 Releases & Pre-training Data

TTM has evolved through several releases, with each version being trained on progressively larger datasets:
* **TTM R1:** Pre-trained on ~250 million samples.
* **TTM R2:** Pre-trained on a larger corpus of ~700 million samples, showing a >15% performance improvement over R1 in standard benchmarks.
* **TTM R2.1:** The latest release, trained on an even larger dataset of ~1 billion samples, with added support for daily and weekly resolutions.

A full list of available models can be found [here](https://github.com/ibm-granite/granite-tsfm/blob/main/tsfm_public/resources/model_paths_config/ttm.yaml).