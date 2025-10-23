from __future__ import annotations

from typing import Dict, Any, List, Tuple
import time
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from .models import Candidate
from .metrics import primary_metric
from .memory import load_memory, get_theory


def _cv_split(task: str, y: np.ndarray):
    if task == "classification":
        return StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    return KFold(n_splits=3, shuffle=True, random_state=42)


def _lookup_tradeoffs(model_id: str, memory: Dict[str, Any]) -> Tuple[float, float]:
    # Returns (interpretabilidad, costo_computacional) in [0,1]
    for m in memory.get("teoria", []):
        if m.get("model_id") == model_id:
            ts = m.get("tradeoff_scores", {}) or {}
            return float(ts.get("interpretabilidad", 0.5)), float(ts.get("costo_computacional", 0.5))
    # Fallback heuristics
    lower = model_id.lower()
    if "logistic" in lower or "linear" in lower:
        return 1.0, 1.0
    if "naive" in lower or "gaussian" in lower:
        return 0.7, 1.0
    if "forest" in lower:
        return 0.3, 0.5
    return 0.5, 0.5


def evaluate_with_j(
    df: pd.DataFrame,
    plan: Dict[str, Any],
    candidates: List[Candidate],
    memory: Dict[str, Any],
) -> Dict[str, Any]:
    task = plan.get("task", "classification")
    target = plan.get("target")
    assert target in df.columns, f"target '{target}' not found in dataframe"
    y = df[target].values
    X = df.drop(columns=[target])
    cv = _cv_split(task, y)

    # Weights for J
    w = {
        "interpretabilidad": 0.3,
        "precision_predictiva": 0.6,
        "costo_computacional": 0.1,
    }
    w.update(plan.get("weights", {}))

    perf_values: List[float] = []
    raw_records: List[Dict[str, Any]] = []

    for cand in candidates:
        cv_scores: List[float] = []
        fit_times: List[float] = []
        for tr, va in cv.split(X, y if task == "classification" else None):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y[tr], y[va]
            model = cand.model
            start = time.time()
            model.fit(Xtr, ytr)
            fit_times.append(time.time() - start)
            if task == "classification":
                prob = model.predict_proba(Xva)[:, 1]
                s = primary_metric(task, yva, None, prob, name="roc_auc")  # higher better
            else:
                pred = model.predict(Xva)
                s = -primary_metric(task, yva, pred, None, name="mae")  # use negative MAE so higher is better
            cv_scores.append(float(s))
        perf = float(np.mean(cv_scores))
        perf_values.append(perf)
        raw_records.append({
            "name": cand.name,
            "perf_raw": perf,
            "fit_time_s": float(np.mean(fit_times)),
        })

    # Normalize performance to [0,1] where higher better
    perf_arr = np.array(perf_values)
    if task == "classification":
        perf_norm = perf_arr  # already in [0,1]
    else:
        # perf was -MAE; shift to [0,1]
        perf_norm = (perf_arr - perf_arr.min()) / (perf_arr.ptp() + 1e-9)

    leaderboard: List[Dict[str, Any]] = []
    for rec, pscore in zip(raw_records, perf_norm):
        interp_score, cost_score = _lookup_tradeoffs(rec["name"], memory)
        # Convert scores to losses
        perf_loss = float(1.0 - pscore)
        interp_loss = float(1.0 - interp_score)
        cost_loss = float(1.0 - cost_score)
        J = (
            w.get("precision_predictiva", 0.6) * perf_loss
            + w.get("interpretabilidad", 0.3) * interp_loss
            + w.get("costo_computacional", 0.1) * cost_loss
        )
        leaderboard.append({
            "name": rec["name"],
            "perf_score": float(pscore),
            "interpretabilidad": interp_score,
            "costo_computacional": cost_score,
            "fit_time_s": rec["fit_time_s"],
            "J": float(J),
            "components": {
                "perf_loss": perf_loss,
                "interp_loss": interp_loss,
                "cost_loss": cost_loss,
                "weights": w,
            },
        })

    leaderboard = sorted(leaderboard, key=lambda r: r["J"])  # minimize J
    return {"leaderboard": leaderboard, "winner": leaderboard[0]}


# --- HPO and Assumption Checks ---

def _param_grid_for(name: str, task: str) -> List[Dict[str, Any]]:
    lower = name.lower()
    grid: List[Dict[str, Any]] = []
    if task == "classification":
        if "logistic" in lower:
            grid = [
                {"est__C": 0.1},
                {"est__C": 1.0},
                {"est__C": 3.0},
            ]
        elif "forest" in lower:
            grid = [
                {"est__n_estimators": 100, "est__max_depth": None},
                {"est__n_estimators": 300, "est__max_depth": 8},
                {"est__n_estimators": 300, "est__max_depth": 16},
            ]
        elif "gaussian" in lower or "naive" in lower:
            grid = [
                {"est__var_smoothing": 1e-9},
                {"est__var_smoothing": 1e-8},
            ]
    else:
        if "linear" in lower:
            grid = [{}]
        elif "forest" in lower:
            grid = [
                {"est__n_estimators": 200, "est__max_depth": None},
                {"est__n_estimators": 400, "est__max_depth": 12},
            ]
    return grid or [{}]


def _assumption_checks(df: pd.DataFrame, target: str, name: str) -> Tuple[Dict[str, Any], float]:
    X = df.drop(columns=[target])
    numeric_cols = list(X.select_dtypes(include=["number"]).columns)
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    checks: Dict[str, Any] = {}
    checks["missing_fraction"] = float(X.isna().mean().mean())
    checks["num_numeric_cols"] = len(numeric_cols)
    checks["num_categorical_cols"] = len(cat_cols)
    checks["high_cardinality_cols"] = [c for c in cat_cols if X[c].nunique(dropna=True) > 100]

    penalty = 0.0
    lower = name.lower()
    if ("logistic" in lower or "linear" in lower) and checks["num_numeric_cols"] > 0:
        # scaling assumed handled by pipeline; no penalty
        penalty += 0.0
    if ("gaussian" in lower) and checks["num_numeric_cols"] == 0:
        penalty += 0.05
    if len(checks["high_cardinality_cols"]) > 0:
        penalty += 0.05
    if checks["missing_fraction"] > 0.3:
        penalty += 0.05
    return checks, float(min(penalty, 0.15))


def evaluate_with_j_hpo_assumptions(
    df: pd.DataFrame,
    plan: Dict[str, Any],
    candidates: List[Candidate],
    memory: Dict[str, Any],
    hpo_budget: int = 6,
) -> Dict[str, Any]:
    task = plan.get("task", "classification")
    target = plan.get("target")
    assert target in df.columns, f"target '{target}' not found in dataframe"
    y = df[target].values
    X = df.drop(columns=[target])
    cv = _cv_split(task, y)

    # Weights for J (includes optional assumption penalty)
    w = {
        "interpretabilidad": 0.3,
        "precision_predictiva": 0.6,
        "costo_computacional": 0.1,
        "assumption_penalty": 0.05,
    }
    w.update(plan.get("weights", {}))

    leaderboard: List[Dict[str, Any]] = []

    for cand in candidates:
        grid = _param_grid_for(cand.name, task)[: max(1, hpo_budget)]
        best_rec: Dict[str, Any] | None = None
        best_J = float("inf")

        checks, assumption_pen = _assumption_checks(df, target, cand.name)
        interp_score, cost_score = _lookup_tradeoffs(cand.name, memory)

        for params in grid:
            cv_scores: List[float] = []
            fit_times: List[float] = []
            model = cand.model.set_params(**params)
            for tr, va in cv.split(X, y if task == "classification" else None):
                Xtr, Xva = X.iloc[tr], X.iloc[va]
                ytr, yva = y[tr], y[va]
                start = time.time()
                model.fit(Xtr, ytr)
                fit_times.append(time.time() - start)
                if task == "classification":
                    prob = model.predict_proba(Xva)[:, 1]
                    s = primary_metric(task, yva, None, prob, name="roc_auc")  # higher better
                else:
                    pred = model.predict(Xva)
                    s = -primary_metric(task, yva, pred, None, name="mae")  # higher better
                cv_scores.append(float(s))
            perf = float(np.mean(cv_scores))
            perf_norm = perf if task == "classification" else float((perf - min(cv_scores)) / (max(cv_scores) - min(cv_scores) + 1e-9))
            perf_loss = float(1.0 - perf_norm)
            interp_loss = float(1.0 - interp_score)
            cost_loss = float(1.0 - cost_score)
            J = (
                w.get("precision_predictiva", 0.6) * perf_loss
                + w.get("interpretabilidad", 0.3) * interp_loss
                + w.get("costo_computacional", 0.1) * cost_loss
                + w.get("assumption_penalty", 0.05) * float(assumption_pen)
            )
            rec = {
                "name": cand.name,
                "params": params,
                "perf_score": perf_norm,
                "interpretabilidad": interp_score,
                "costo_computacional": cost_score,
                "assumptions": checks,
                "assumption_penalty": assumption_pen,
                "fit_time_s": float(np.mean(fit_times)) if fit_times else None,
                "J": float(J),
                "components": {
                    "perf_loss": perf_loss,
                    "interp_loss": interp_loss,
                    "cost_loss": cost_loss,
                    "assumption_penalty": assumption_pen,
                    "weights": w,
                },
            }
            if J < best_J:
                best_J = J
                best_rec = rec
        assert best_rec is not None
        leaderboard.append(best_rec)

    leaderboard = sorted(leaderboard, key=lambda r: r["J"])  # minimize J
    return {"leaderboard": leaderboard, "winner": leaderboard[0]}
