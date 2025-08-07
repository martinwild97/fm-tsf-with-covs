# MOIRAI

| | |
| :--- | :--- |
| **Developer** | Salesforce AI Research |
| **Paper** | [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592) |
| **Code/Model** | [GitHub](https://github.com/SalesforceAIResearch/uni2ts), [Hugging Face](hhttps://huggingface.co/collections/Salesforce/moirai-r-models-65c8d3a94c51428c300e0742) |

---

## 🧠 Core Idea & Architecture

MOIRAI (Masked Encoder-based Universal Time Series Forecasting Transformer) is a foundation model designed to be a universal forecaster, capable of handling a wide variety of time series without task-specific retraining. It was pre-trained on the **LOTSA (Large-scale Open Time Series Archive)**, a vast and diverse corpus of time series data that has also become a foundational dataset for other models.

Its core architecture is a **masked encoder-only Transformer**.
* **Pre-training:** During pre-training, it uses a masking strategy similar to BERT. Random patches of the input time series are masked, and the model is trained to reconstruct the original values. This self-supervised task allows it to learn rich, general-purpose representations of temporal patterns.
* **Input Processing:** It uses **patching** to segment time series and incorporates learnable **Variate ID Embeddings** to distinguish between different channels in a multivariate input. It also supports dynamic patch sizing to better handle time series of different granularities.

A **MOIRAI-MoE (Mixture-of-Experts)** variant also exists, which replaces standard feed-forward layers with sparse MoE layers for more data-driven specialization and training efficiency.

---

## ✨ Special Features & Data Handling

* **Unified Multivariate & Covariate Integration:** MOIRAI's standout feature is its "homogenized input" strategy. It treats all time series—both targets and exogenous variables—as interchangeable channels or "variates" in a single input tensor.
* **Any-variate Attention (AVA):** This innovative mechanism allows the model to handle an arbitrary number of variables. In its "channel-mixed" mode, AVA computes attention across all patches from all variates, enabling it to learn the complex, non-linear interactions between targets and covariates directly.
* **Advanced Probabilistic Forecasting:** MOIRAI provides probabilistic outputs by predicting the parameters of a **mixture distribution** (including Student's t, log-normal, and others). This allows it to model a wide range of uncertainties and data shapes, making it one of the few models to offer this for complex multivariate scenarios.

---

## 👍 Strengths

* **Truly Universal Input:** It is one of the very few models that supports probabilistic, multivariate, and covariate-informed forecasting out of the box within a single, unified architecture.
* **Proven Fine-Tuning Potential:** The true power of MOIRAI is unlocked through fine-tuning. A fine-tuned version of the model **VN1 weeky demand forecasting challange**, making it a highly recommended model *if* you plan to adapt it to a specific dataset.
* **Deep Covariate Integration:** The AVA mechanism allows for a deep, end-to-end learning of relationships between all variables, which is theoretically more powerful than modular or external approaches.

---

## 👎 Weaknesses & Practical Tips

* **Unreliable Covariate Impact:** In practice, the benefit of adding covariates in a zero-shot setting is **highly inconsistent**. Because the model was pre-trained by randomly sampling different time series together, it did not explicitly learn structured covariate relationships. As a result, adding external variables can either improve or significantly degrade forecast accuracy, making it a high-risk choice without careful, task-specific validation or fine-tuning.
* **VRAM Issues with High Dimensionality:** The model struggles with high-dimensional inputs. Providing a long context length or a large number of variables (targets + covariates) can easily lead to **CUDA out-of-memory errors**.
* **MOIRAI-MoE Not Recommended:** The Mixture-of-Experts variant consistently underperformed in personal tests, especially with multivariate or covariate data. Seems that the MoE variant is only working on unique datasets.
* **Difficult to Fine-Tune:** While fine-tuning unlocks the model's best performance, the process is not straightforward and can be challenging to implement correctly.
* **Outdated Dependencies:** The underlying `uni2ts` library officially supports a specific version of PyTorch (e.g., 2.1) which has known security vulnerabilities. While not necessarily exploited by the model, this creates a maintenance and security concern for production environments.

---

## 📏 Available Model Sizes

MOIRAI is available in several sizes for the standard, 1.1, and MoE versions. The 1.1. version achieves significant improvements for low-frequency cases like Yearly and Quarterly data.

| Model Name | Parameters |
| :--- | :--- |
| **moirai-1.0-R-small** | 14M |
| **moirai-1.0-R-base** | 91M |
| **moirai-1.0-R-large** | 311M |
| **moirai-1.1-R-small** | 14M |
| **moirai-1.1-R-base** | 91M |
| **moirai-1.1-R-large** | 311M |
| **moirai-moe-1.0-R-small** | 11M (activated) |
| **moirai-moe-1.0-R-base** | 86M (activated) |