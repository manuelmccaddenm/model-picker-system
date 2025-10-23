# Agent Communication Contracts

This document defines the strict JSON contracts between agents to ensure clean handoffs.

---

## 1. Data Agent → Model Agent

### Contract: `BusinessContext`

The Data Agent MUST provide this complete structure to the Model Agent:

```json
{
  "task": "string (REQUIRED) - Business problem description, e.g., 'Predict customer churn'",
  "target": "string (REQUIRED) - Target column name, e.g., 'Churn' or 'target'",
  "business_rules": [
    "string - User preferences and constraints",
    "e.g., 'User prioritizes interpretability over accuracy'",
    "e.g., 'Exclude column X due to data leakage'"
  ],
  "objective": "'_classify' | '_regress' (REQUIRED) - Inferred from target type",
  "primary_metric": "'_accuracy' | '_rmse' | '_loss' (REQUIRED) - Inferred from objective",
  "clarifying_questions": [],
  "dataset_profile": {
    "n_rows": "int - Total rows in dataset",
    "n_cols": "int - Total columns",
    "columns": [
      {
        "name": "string - Column name",
        "dtype": "string - pandas dtype",
        "non_null": "int - Non-null count",
        "nulls": "int - Null count",
        "distinct": "int - Unique values",
        "example_values": ["string", "..."],
        "is_numeric": "bool",
        "is_categorical": "bool",
        "is_datetime": "bool"
      }
    ]
  },
  "dataset_inferences": {
    "dataset_kind": "'tabular' | 'time_series' | 'panel' | 'unknown'",
    "timeseries": {
      "time_col": "string | null",
      "inferred_frequency": "string | null",
      "n_series": "int | null"
    },
    "panel": {
      "entity_id_col": "string | null",
      "time_col": "string | null"
    },
    "task_suggestion": {
      "inferred_target_type": "string - e.g., 'binary', 'multiclass', 'continuous'",
      "suggested_objective": "'_classify' | '_regress' | null",
      "suggested_primary_metric": "'_accuracy' | '_rmse' | '_loss' | null",
      "class_imbalance_flag": "bool | null"
    },
    "data_warnings": ["string - Data quality issues"]
  },
  "key_insights": [
    "string - Actionable insight about the data",
    "e.g., 'Target is binary with 12% positive class - imbalanced'",
    "e.g., 'Feature X has 40% missing values'"
  ]
}
```

### Required Fields for Handoff

The Model Agent **CANNOT** proceed without:
- ✅ `task` (not null/empty)
- ✅ `target` (not null/empty)
- ✅ `objective` (not null)
- ✅ `primary_metric` (not null)
- ✅ `dataset_profile` (not null)

Optional but recommended:
- `business_rules` (helps with J definition)
- `key_insights` (helps with model selection)

---

## 2. Model Agent → Eval Agent

### Contract: `ModelProposals`

The Model Agent MUST provide:

```json
{
  "J_definition": "string (REQUIRED) - Formula, e.g., 'J = 0.5*interpretabilidad + 0.4*precision_predictiva + 0.1*costo_computacional'",
  "model_specs": [
    {
      "model_id": "string (REQUIRED) - Unique identifier, e.g., 'LogisticRegression'",
      "rationale": "string (REQUIRED) - Why this model, 1-3 sentences",
      "implementation": {
        "library": "string (REQUIRED) - 'scikit-learn' | 'prophet' | 'statsmodels'",
        "estimator_class": "string (REQUIRED) - e.g., 'LogisticRegression'",
        "preprocessing_steps": ["string - e.g., 'StandardScaler'"],
        "hyperparameters": "string (REQUIRED) - JSON string, e.g., '{\"C\": 1.0}'",
        "notes": "string - Implementation notes"
      }
    }
  ]
}
```

### Required Fields

- ✅ Exactly 3 `model_specs` (no more, no less)
- ✅ Each spec must have `model_id`, `rationale`, `implementation`
- ✅ `J_definition` must be a valid formula string
- ✅ `hyperparameters` must be valid JSON string (not object!)

---

## 3. Eval Agent → Eval Agent (Rapid Loop)

### Contract: `RapidLoopResult`

```json
{
  "iterations": [
    {
      "evaluation": "string - Analysis of current state",
      "mutations": [
        "string - Concrete next step",
        "e.g., 'Try LogisticRegression with C=0.5'"
      ]
    }
  ],
  "final_recommendation": {
    "model_id": "string (REQUIRED)",
    "rationale": "string (REQUIRED)",
    "implementation": {
      "library": "string",
      "estimator_class": "string",
      "preprocessing_steps": ["string"],
      "hyperparameters": "string (JSON)",
      "notes": "string"
    }
  }
}
```

---

## 4. Eval Agent → Memory (Slow Loop)

### Contract: `SlowLoopResult`

```json
{
  "learnings": [
    "string - Generalizable lesson",
    "e.g., 'For imbalanced binary classification, tree-based models outperformed linear models'"
  ],
  "suggested_memory_updates": {
    "new_experiences": [
      {
        "task": "string - Task type",
        "dataset": "string - Dataset name",
        "outcome": "string - What worked/didn't work",
        "learnings": "string - Key takeaways"
      }
    ],
    "new_lessons": ["string - Analyst lessons"],
    "notes": "string - Additional context"
  }
}
```

---

## Validation Rules

### Data Agent Output Validation

Before handoff, check:
```python
assert bc.task is not None and bc.task != "", "task must be set"
assert bc.target is not None and bc.target != "", "target must be set"
assert bc.objective in ["_classify", "_regress"], "objective must be valid"
assert bc.primary_metric in ["_accuracy", "_rmse", "_loss"], "metric must be valid"
assert bc.dataset_profile is not None, "dataset_profile must exist"
assert len(bc.clarifying_questions) == 0, "no pending questions"
```

### Model Agent Output Validation

Before handoff, check:
```python
assert mp.J_definition is not None and mp.J_definition != "", "J must be defined"
assert len(mp.model_specs) == 3, "must have exactly 3 models"
for spec in mp.model_specs:
    assert spec.model_id is not None
    assert spec.rationale is not None
    assert spec.implementation is not None
    assert spec.implementation.library in ["scikit-learn", "prophet", "statsmodels"]
    # Validate hyperparameters is valid JSON
    import json
    json.loads(spec.implementation.hyperparameters)
```

---

## Schema Evolution

If you need to change a contract:

1. **Update the Pydantic model** in `src/agents.py`
2. **Update this CONTRACTS.md** document
3. **Update agent instructions** to match new schema
4. **Test the full pipeline** to ensure no breaks

---

## Why This Matters

- ✅ **Type Safety**: Pydantic validates at runtime
- ✅ **Clear Expectations**: Each agent knows exactly what to provide
- ✅ **Easier Debugging**: Schema errors are caught immediately
- ✅ **Documentation**: This file is the source of truth
- ✅ **OpenAI Strict Mode**: Compatible with strict JSON schemas

