from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB


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
