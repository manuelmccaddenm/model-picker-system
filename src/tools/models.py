from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR


@dataclass
class Candidate:
    name: str
    model: object


def _build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    numeric_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", categorical_tf, categorical_cols),
        ]
    )
    return preprocessor


def build_candidates(df: pd.DataFrame, target: str, task: str = "classification") -> List[Candidate]:
    X = df.drop(columns=[target], errors="ignore")
    pre = _build_preprocessor(X)
    cands: List[Candidate] = []

    if task == "classification":
        cands.append(
            Candidate(
                name="LogisticRegression",
                model=Pipeline(steps=[("pre", pre), ("est", LogisticRegression(max_iter=1000))]),
            )
        )
        cands.append(
            Candidate(
                name="RandomForestClassifier",
                model=Pipeline(steps=[("pre", pre), ("est", RandomForestClassifier(random_state=42))]),
            )
        )
        cands.append(
            Candidate(
                name="GaussianNB",
                model=Pipeline(steps=[("pre", pre), ("est", GaussianNB())]),
            )
        )
    else:
        cands.append(
            Candidate(
                name="LinearRegression",
                model=Pipeline(steps=[("pre", pre), ("est", LinearRegression())]),
            )
        )
        cands.append(
            Candidate(
                name="RandomForestRegressor",
                model=Pipeline(steps=[("pre", pre), ("est", RandomForestRegressor(random_state=42))]),
            )
        )
    return cands


def build_candidates_from_proposals(
    df: pd.DataFrame,
    target: str,
    proposals: List[dict],
    task: str = "classification",
    include_fallback: bool = False,
) -> List[Candidate]:
    """Build sklearn candidates from Model Agent proposals, aligned with our pipelines.

    Falls back to default candidates if none can be built. Optionally appends a small
    diverse set when include_fallback=True to broaden search under low confidence.
    """
    X = df.drop(columns=[target], errors="ignore")
    pre = _build_preprocessor(X)
    candidates: List[Candidate] = []

    # Map estimator name → class
    est_map = {
        "LogisticRegression": LogisticRegression,
        "LinearRegression": LinearRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "GaussianNB": GaussianNB,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "SVC": SVC,
        "SVR": SVR,
    }

    for spec in proposals:
        try:
            model_id = spec.get("model_id", "Unknown")
            impl = spec.get("implementation", {}) or {}
            est_name = impl.get("estimator_class") or (
                "LogisticRegression" if task == "classification" else "LinearRegression"
            )
            Estimator = est_map.get(est_name)
            if Estimator is None:
                # Skip unknown estimators
                continue
            # Parse hyperparameters from JSON string if present
            hyper_str = impl.get("hyperparameters", "{}")
            try:
                import json
                hyper = json.loads(hyper_str)
            except Exception:
                hyper = {}
            # Remove deprecated/unsafe args
            if "multi_class" in hyper:
                hyper.pop("multi_class", None)
            est = Estimator(**hyper)
            model = Pipeline([("pre", pre), ("est", est)])
            candidates.append(Candidate(name=model_id, model=model))
        except Exception as e:
            print(f"Warning: could not build candidate for {spec.get('model_id', 'Unknown')}: {e}")
            continue

    if include_fallback:
        try:
            if task == "classification":
                candidates.append(Candidate(
                    name="GradientBoostingClassifier",
                    model=Pipeline([("pre", pre), ("est", GradientBoostingClassifier())])
                ))
                candidates.append(Candidate(
                    name="SVC",
                    model=Pipeline([("pre", pre), ("est", SVC(probability=True))])
                ))
            else:
                candidates.append(Candidate(
                    name="GradientBoostingRegressor",
                    model=Pipeline([("pre", pre), ("est", GradientBoostingRegressor())])
                ))
                candidates.append(Candidate(
                    name="SVR",
                    model=Pipeline([("pre", pre), ("est", SVR())])
                ))
        except Exception as e:
            print(f"Warning: could not append fallback candidates: {e}")

    if not candidates:
        print("Warning: no candidates from proposals; using defaults")
        return build_candidates(df, target=target, task=task)
    return candidates
