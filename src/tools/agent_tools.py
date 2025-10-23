from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
from agents import function_tool
from pathlib import Path
from .memory import load_memory, get_theory, get_experiences
from .models import build_candidates
from .evaluator import evaluate_with_j_hpo_assumptions


@function_tool
def memory_get_theory(task: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Return model theory entries filtered by task (and limited)."""
    base = Path(__file__).resolve().parents[2]
    mem = load_memory(base / "memory" / "memory.json")
    return get_theory(mem, task=task, limit=limit)


@function_tool
def memory_get_experiences(task: str, k: int = 5) -> List[Dict[str, Any]]:
    """Return up to k experiences for a given task."""
    base = Path(__file__).resolve().parents[2]
    mem = load_memory(base / "memory" / "memory.json")
    return get_experiences(mem, task=task, k=k)


@function_tool
def perform_eda(dataset_path: str, sample_rows: int = 1000) -> Dict[str, Any]:
    """Comprehensive EDA tool that profiles a CSV dataset.
    
    This tool acts like a Code Interpreter - it loads the data, computes statistics,
    detects patterns, and provides insights to help the LLM understand the dataset.
    
    Returns a rich profile including:
    - Basic stats (rows, columns, dtypes)
    - Column-level analysis (nulls, cardinality, types)
    - Pattern detection (time series, panel data)
    - Data quality warnings
    """
    df = pd.read_csv(dataset_path)
    original_rows = len(df)
    
    # Sample if too large
    if sample_rows and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=42)
    
    # Column-level profiling
    cols: List[Dict[str, Any]] = []
    datetime_cols = []
    potential_id_cols = []
    high_cardinality_cols = []
    high_null_cols = []
    
    for c in df.columns:
        s = df[c]
        null_pct = s.isna().sum() / len(s)
        distinct = s.nunique(dropna=True)
        
        # Detect column types
        is_numeric = pd.api.types.is_numeric_dtype(s)
        is_datetime = pd.api.types.is_datetime64_any_dtype(s)
        is_categorical = distinct <= 20 and not is_numeric
        
        # Try to parse as datetime if string
        if not is_datetime and s.dtype == 'object':
            try:
                pd.to_datetime(s.dropna().head(100))
                is_datetime = True
                datetime_cols.append(c)
            except:
                pass
        elif is_datetime:
            datetime_cols.append(c)
        
        # Detect potential ID columns (high cardinality, mostly unique)
        if distinct / len(s.dropna()) > 0.95 and distinct > 100:
            potential_id_cols.append(c)
        
        # Flag issues
        if distinct > 100 and not is_numeric:
            high_cardinality_cols.append(c)
        if null_pct > 0.3:
            high_null_cols.append(c)
        
        cols.append({
            "name": c,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "nulls": int(s.isna().sum()),
            "null_pct": float(null_pct),
            "distinct": int(distinct),
            "example_values": [str(v) for v in s.dropna().unique()[:5]],
            "is_numeric": is_numeric,
            "is_categorical": is_categorical,
            "is_datetime": is_datetime,
        })
    
    # Detect dataset patterns
    dataset_kind = "tabular"
    timeseries_info = None
    panel_info = None
    
    if datetime_cols:
        # Could be time series or panel
        if len(potential_id_cols) > 0:
            dataset_kind = "panel"
            panel_info = {
                "potential_entity_cols": potential_id_cols,
                "potential_time_cols": datetime_cols,
            }
        else:
            dataset_kind = "time_series"
            timeseries_info = {
                "time_cols": datetime_cols,
                "n_series": 1,
            }
    
    # Generate warnings
    warnings = []
    if high_null_cols:
        warnings.append(f"High missing data (>30%) in columns: {', '.join(high_null_cols)}")
    if high_cardinality_cols:
        warnings.append(f"High cardinality (>100 unique) in categorical columns: {', '.join(high_cardinality_cols)}")
    if len(df.columns) > 100:
        warnings.append(f"High dimensionality: {len(df.columns)} columns")
    if original_rows != len(df):
        warnings.append(f"Dataset sampled: showing {len(df)} of {original_rows} rows")
    
    return {
        "n_rows": int(original_rows),
        "n_rows_sampled": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": cols,
        "dataset_kind": dataset_kind,
        "datetime_cols": datetime_cols,
        "potential_id_cols": potential_id_cols,
        "timeseries_info": timeseries_info,
        "panel_info": panel_info,
        "warnings": warnings,
    }


@function_tool
def evaluate_models_j(
    dataset_path: str,
    target: str,
    task: str,
    w_perf: float = 0.6,
    w_interp: float = 0.3,
    w_cost: float = 0.1,
    hpo_budget: int = 4,
) -> Dict[str, Any]:
    """Evaluate baseline sklearn candidates on the dataset and return leaderboard minimizing J.
    Args:
      dataset_path: absolute or relative file path to CSV
      target: target column name
      task: 'classification' or 'regression'
      w_perf, w_interp, w_cost: weights for J components
      hpo_budget: HPO trials per candidate
    Returns:
      Dict with keys: leaderboard (list), winner (dict)
    """
    df = pd.read_csv(dataset_path)
    base = Path(__file__).resolve().parents[2]
    mem = load_memory(base / "memory" / "memory.json")
    plan = {
        "task": task,
        "target": target,
        "weights": {
            "precision_predictiva": float(w_perf),
            "interpretabilidad": float(w_interp),
            "costo_computacional": float(w_cost),
        },
    }
    cands = build_candidates(df, target=target, task=task)
    return evaluate_with_j_hpo_assumptions(df, plan, cands, mem, hpo_budget=hpo_budget)
