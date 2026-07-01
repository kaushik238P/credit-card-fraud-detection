# Validation Report: `archive.zip`

| Field | Value |
|-------|-------|
| **Status** | ❌ FAILED |
| Report ID | `8fbd8db0-865a-40b4-9649-ead131cda440` |
| Generated | `2026-06-30T01:51:39Z` |
| Schema | `fraud_transaction_v1` v`1.0.0` |
| Dataset Hash | `4e32829b9ba5a6b17af707c513c15204011044df0e8971e431cedca3d8c0a8a1` |
| Duration | `4939.4 ms` |

## Summary

| Metric | Value |
|--------|-------|
| Total Checks | 8 |
| Passed | 6 |
| Warnings | 1 |
| Errors | 1 |
| Overall | ❌ FAILED |

## Check Results

| # | Check | Category | Status | Severity | Time (ms) | Message |
|---|-------|----------|--------|----------|-----------|---------|
| 1 | `check_empty_dataframe` | STRUCTURAL | ✅ PASSED | ℹ️ INFO | 0.0 | DataFrame is non-empty: 1,296,675 rows x 23 columns. |
| 2 | `check_schema` | STRUCTURAL | ✅ PASSED | ℹ️ INFO | 0.2 | Schema validation passed: all 23 required column(s) present, column count 23 ... |
| 3 | `check_data_types` | STRUCTURAL | ❌ FAILED | 🔴 ERROR | 3.6 | Data type mismatch in 12 column(s): ['trans_date_trans_time', 'merchant', 'ca... |
| 4 | `check_empty_columns` | STRUCTURAL | ✅ PASSED | ℹ️ INFO | 38.5 | No entirely-null columns detected. |
| 5 | `check_missing_values` | QUALITY | ✅ PASSED | ℹ️ INFO | 135.4 | No missing values detected across all columns. |
| 6 | `check_duplicate_rows` | QUALITY | ✅ PASSED | ℹ️ INFO | 4696.6 | No duplicate rows detected in 1,296,675 rows. |
| 7 | `check_target_column` | TARGET | ✅ PASSED | ℹ️ INFO | 27.7 | Target column 'is_fraud' is valid: 0 null(s), 2 unique class(es). |
| 8 | `check_class_distribution` | TARGET | ⚠️ WARNING | ⚠️ WARNING | 36.2 | Class imbalance detected in 'is_fraud': minority class '1' = 0.5789% (7,506 s... |

## Issues

### ❌ Errors (1)

**`check_data_types`** &nbsp;·&nbsp; STRUCTURAL

> Data type mismatch in 12 column(s): ['trans_date_trans_time', 'merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'dob', 'trans_num'].

<details>
<summary>Details</summary>

```json
{
  "columns_checked": 23,
  "mismatches": [
    {
      "column": "trans_date_trans_time",
      "expected_category": "categorical",
      "actual_dtype": "str",
      "actual_category": "unknown"
    },
    {
      "column": "merchant",
      "expected_category": "categorical",
      "actual_dtype": "str",
      "actual_category": "unknown"
    },
    {
      "column": "category",
      "expected_category": "categorical",
      "actual_dtype": "str",
      "actual_category": "unknown"
    },
    {
      "column": "first",
      "expected_category": "categorical",
      "actual_dtype": "str",
      "actual_category": "unknown"
    },
    {
      "column": "last",
      "expected_category": "categorical",
      "actual_dtype": "str",
      "actual_category": "unknown"
    },
    {
      "column": "gender",
      "expected_category": "categorical",
      "actual_dtype": "str",
      "actual_category": "unknown"
    },
    {
      "column": "street",
      "expected_category": "categorical",
      "actual_dtype": "str",
      "actual_category": "unknown"
    },
    {
      "column": "city",
      "expected_category": "categorical",
      "actual_dtype": "str",
      "actual_category": "unknown"
    },
    {
      "column": "state",
      "expected_category": "categorical",
      "actual_dtype": "str",
      "actual_category": "unknown"
    },
    {
      "column": "job",
      "expected_category": "categorical",
      "actual_dtype": "str",
      "actual_category": "unknown"
    },
    {
      "column": "dob",
      "expected_category": "categorical",
      "actual_dtype": "str",
      "actual_category": "unknown"
    },
    {
      "column": "trans_num",
      "expected_category": "categorical",
      "actual_dtype": "str",
      "actual_category": "unknown"
    }
  ],
  "mismatch_count": 12
}
```

</details>

### ⚠️ Warnings (1)

**`check_class_distribution`** &nbsp;·&nbsp; TARGET

> Class imbalance detected in 'is_fraud': minority class '1' = 0.5789% (7,506 samples). Imbalance ratio: 171.8:1. Consider SMOTE, class weights, or threshold tuning.

<details>
<summary>Details</summary>

```json
{
  "target_column": "is_fraud",
  "class_counts": {
    "0": 1289169,
    "1": 7506
  },
  "class_percentages": {
    "0": 99.4211,
    "1": 0.5789
  },
  "total_labelled_rows": 1296675,
  "minority_class": "1",
  "minority_count": 7506,
  "minority_pct": 0.5789,
  "majority_class": "0",
  "majority_count": 1289169,
  "imbalance_ratio": 171.75,
  "warning_threshold_pct": 5.0
}
```

</details>

## Metadata Snapshot

```json
{
  "file_name": "archive.zip",
  "file_path": "D:\\credit-card-fraud-detection\\data\\raw\\archive.zip",
  "file_extension": ".zip",
  "file_size_bytes": 211766662,
  "num_rows": 1296675,
  "num_columns": 23,
  "column_names": [
    "Unnamed: 0",
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud"
  ],
  "memory_usage_bytes": 447299402,
  "target_column": "is_fraud",
  "load_timestamp": "2026-06-30T01:51:34.890715+00:00",
  "dataset_hash": "4e32829b9ba5a6b17af707c513c15204011044df0e8971e431cedca3d8c0a8a1",
  "dataset_version": null,
  "loaded_by": null
}
```
