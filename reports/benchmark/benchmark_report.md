# Model Performance Benchmark Report

Generated at: `2026-06-30T11:24:42.698754+00:00`

## Executive Summary

> [!IMPORTANT]
> **Best Model Recommendation**  
> The recommended model is **CATBOOST**, achieving a primary metric **PR_AUC of 0.9309** and an **F1 score of 0.8691**.  
> This model is promoted to stage **Production** in the MLflow Model Registry.

### Quick Stats
- **Dataset Evaluated**: `Fraud Transaction Dataset`
- **Dataset Version**: `1.0.0`
- **Successful Runs**: `5`
- **Failed Runs**: `0`
- **Fastest Training**: `LOGISTIC_REGRESSION`
- **Lowest False Positive Rate**: `CATBOOST`

## Model Performance Ranking

| Rank | Model Name | Status | PR_AUC | F1 | Recall | Precision | FPR | Training Time | MLflow ID |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **CATBOOST** | `SUCCESS` | 0.9309 | 0.8691 | 0.8177 | 0.9275 | 0.0004 | 72.91s | `fc32ef80` |
| 2 | **RANDOM_FOREST** | `SUCCESS` | 0.9174 | 0.8719 | 0.8223 | 0.9279 | 0.0004 | 262.42s | `620bc60d` |
| 3 | **LIGHTGBM** | `SUCCESS` | 0.9088 | 0.8392 | 0.7700 | 0.9221 | 0.0004 | 38.43s | `3ee157aa` |
| 4 | **XGBOOST** | `SUCCESS` | 0.9045 | 0.8338 | 0.7689 | 0.9106 | 0.0005 | 40.13s | `ec4ab855` |
| 5 | **LOGISTIC_REGRESSION** | `SUCCESS` | 0.0074 | 0.0130 | 1.0000 | 0.0065 | 0.9975 | 10.13s | `454d1b2c` |

## Detailed Metric Comparison

| Model Name | ROC_AUC | Specificity | Balanced Accuracy | MCC | Alert Rate | Precision@K | Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CATBOOST** | 0.9976 | 0.9996 | 0.9086 | 0.8701 | 0.0057 | 1.0000 | 0.4047 |
| **RANDOM_FOREST** | 0.9958 | 0.9996 | 0.9109 | 0.8727 | 0.0058 | 1.0000 | 0.3202 |
| **LIGHTGBM** | 0.9971 | 0.9996 | 0.8848 | 0.8417 | 0.0054 | 1.0000 | 0.9043 |
| **XGBOOST** | 0.9981 | 0.9995 | 0.8842 | 0.8358 | 0.0055 | 1.0000 | 0.9523 |
| **LOGISTIC_REGRESSION** | 0.5315 | 0.0025 | 0.5013 | 0.0040 | 0.9975 | 0.0000 | 0.0013 |

## Benchmark Execution Timeline

| Step Name | Start Time | End Time | Duration (Seconds) |
| :--- | :--- | :--- | :--- |
| Dataset Ready | `2026-06-30T11:15:33.693664+00:00` | `2026-06-30T11:16:27.427439+00:00` | 53.7371s |
| Model Discovery | `2026-06-30T11:16:27.427439+00:00` | `2026-06-30T11:16:27.427439+00:00` | 0.0002s |
| Training | `2026-06-30T11:16:27.427439+00:00` | `2026-06-30T11:24:42.698235+00:00` | 424.0192s |
| Evaluation | `2026-06-30T11:16:27.427439+00:00` | `2026-06-30T11:24:42.698235+00:00` | 40.0014s |
| MLflow Logging | `2026-06-30T11:16:27.427439+00:00` | `2026-06-30T11:24:42.698235+00:00` | 31.2391s |
| Ranking | `2026-06-30T11:24:42.698235+00:00` | `2026-06-30T11:24:42.698235+00:00` | 0.0002s |

## Recommendation Summary

Based on the primary metric **PR_AUC (Precision-Recall Area Under Curve)**, the best performing model is **CATBOOST**.

- It achieves the highest precision-recall balance with a score of **0.9309**.
- The decision threshold was optimized using maximum F1 strategy to **0.4047**.
- This model is fully tracked under MLflow Run ID `fc32ef801696436480549b6198ea5e62`.