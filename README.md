# Foundation Models for Time Series Forecasting with Exogenous Variables: A Survey and Benchmark

**Master Thesis | Fachhochschule Kiel | June 2025**

**Author:** [Martin Wild](https://www.linkedin.com/in/martin-wild-7056582b0)
**First Examiner:** [Prof. Dr. Stephan Doerfel](https://www.linkedin.com/in/doerfel/)
**Second Examiner:** [Dr. Thomas Christ](https://www.linkedin.com/in/thomas-christ-81116319a/)

## 📜 Abstract

Foundation Models (FMs) are emerging as a powerful new paradigm for Time Series Forecasting (TSF), offering the promise of strong generalization capabilities with minimal task-specific engineering. However, while the use of exogenous variables is critical for achieving high accuracy in many real-world forecasting scenarios, their integration into FMs remains a nascent and fragmented research area. This thesis addresses this significant research gap by providing the first systematic overview and comprehensive empirical benchmark of FMs that incorporate exogenous variables.

The study follows a two-part methodology: first, a structured literature survey is conducted to identify and categorize the diverse architectural approaches into two dominant paradigms: Modular / Staged Integration and Unified / Deep Integration. Second, a novel empirical benchmark is executed, comparing selected FMs against established baseline models across a wide range of exogenous variable scenarios.

The key findings reveal that the impact of exogenous variables on FM performance is not universally positive but is highly dependent on both the model’s architecture and the specific characteristics of the dataset. The modular approach using an external regressor, as employed by Chronos, proved to be the most reliable strategy for leveraging external information. Conversely, the study demonstrates that for many FMs, an indiscriminate inclusion of exogenous variables can degrade performance, challenging the "more data is better" axiom and highlighting a deep sensitivity to configuration. The results also identify a critical trade-off: FMs offer significant advantages in computational speed and workflow simplicity but currently suffer from a profound lack of interpretability and model-specific practical issues.

Ultimately, this work concludes that for the practical application of FMs with exogenous variables, the field must likely move beyond a pure "zero-shot" approach toward a more nuanced "pre-train" and "fine-tune" paradigm. By providing a clear categorization of integration methods, a foundational empirical benchmark, and actionable insights into model behavior, this thesis establishes a crucial baseline for future research in a dynamic field whose advancement is currently hampered by a scarcity of suitable public datasets.

## 🔑 Key Findings & Conclusion

  * **The "No Free Lunch" Rule for Covariates:** Adding exogenous variables is a high-risk, architecture-dependent endeavor, not a guaranteed improvement. While some models benefit reliably, others can be significantly degraded by the same data. The method of integration is paramount.

  * **Primacy of Modular Architectures (in the Current Paradigm):** In a near-zero-shot context, modular integration (e.g., using an external regressor with a strong univariate FM like Chronos) is the most robust and practical strategy for leveraging external information.

  * **A Necessary Shift from "Zero-Shot" to "Fine-Tuning":** The generalized knowledge from pre-training is often insufficient to capture complex, task-specific covariate relationships. Effective real-world deployment will require a paradigm shift to a "pre-train and fine-tune" workflow to achieve reliable performance.

  * **The Data Bottleneck:** The primary obstacle to advancing the field is the systemic lack of large-scale, high-quality public benchmark datasets with rich and well-documented covariates. Without such data, the full potential of any architecture remains unrealized.

  * **The FM Trade-Off:** The primary strengths of FMs are operational: significant gains in **computational speed** and a **simplified workflow**. The main weaknesses are a profound **lack of interpretability**, significant model-specific issues (e.g., prohibitive fine-tuning times, failure on certain data patterns), and a dependency on future-known covariates for the most successful integration methods.

## 🚀 Code soon to be released

The full codebase, including the benchmark framework and the modular Foundation Model Wrapper API, will be made publicly available soon to promote reproducible research.

## 🎓 Citing This Work

If you find this work useful for your research, please consider citing the thesis:

```bibtex
@mastersthesis{wild2025,
  author        = {Martin Wild},
  title         = {Foundation Models for Time Series Forecasting with Exogenous Variables: A Survey and Benchmark},
  school        = {Fachhochschule Kiel},
  year          = {2025},
  month         = jun,
  address       = {Kiel, Germany},
  note          = {Department of Media},
  url           = {https://github.com/martinwild97/fm-tsf-with-covs}
}
```

## 🤝 How to Contribute

Contributions, issues, and feature requests are welcome\!

## 📄 License

This project is licensed under the **MIT License**.
