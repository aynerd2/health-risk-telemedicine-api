"""
Train the diabetes risk model (Sections 3.7-3.10) on
backend/training_data/Diabetes.csv (100,000 rows, heavily imbalanced target:
~8,500 positive / ~91,500 negative — 8.5%).

class_weight="balanced" reweights the loss inversely proportional to class
frequency so the model doesn't just learn to always predict "no diabetes";
recall is reported alongside accuracy/precision/F1 since, per Section 3.10,
missing an at-risk patient (a false negative) is the costlier error for a
screening tool.

    python scripts/train_diabetes_model.py
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_common import run_training

DATA_PATH = Path(__file__).resolve().parent.parent / "training_data" / "Diabetes.csv"

NUMERIC_FEATURES = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
CATEGORICAL_FEATURES = ["gender", "hypertension", "heart_disease", "smoking_history"]
TARGET = "diabetes"


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    # 3,854 exact duplicate rows. Caveat: 99% of them have bmi == 27.32 (the
    # dataset's fill value for missing BMI) and HbA1c/glucose take only 18
    # values each, so most are probably distinct patients colliding on coarse
    # values rather than copied records. Dropped anyway so no identical row
    # can sit on both sides of the split.
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {n_raw - len(df)} exact duplicate rows ({n_raw} -> {len(df)})")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    run_training("diabetes", build_pipeline(), X, y)


if __name__ == "__main__":
    main()
