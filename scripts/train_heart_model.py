"""
Train the heart-disease risk model (Sections 3.7-3.10) on
backend/training_data/Heart_disease.csv (UCI heart-disease dataset, 1025
rows, target already binary and balanced: 526 disease / 499 no-disease).

Every column in this dataset is already numeric — sex/cp/fbs/restecg/exang/
slope/ca/thal are small-cardinality codes, so they're one-hot encoded as
categoricals; age/trestbps/chol/thalach/oldpeak are continuous and scaled.

    python scripts/train_heart_model.py
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_common import run_training

DATA_PATH = Path(__file__).resolve().parent.parent / "training_data" / "Heart_disease.csv"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"


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
    return Pipeline([("preprocess", preprocessor), ("clf", LogisticRegression(max_iter=1000))])


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    run_training("heart", build_pipeline(), X, y)


if __name__ == "__main__":
    main()
