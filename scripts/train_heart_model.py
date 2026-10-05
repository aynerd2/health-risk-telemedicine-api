"""
Train the heart-disease risk model (Sections 3.7-3.10) on
backend/training_data/Heart_disease.csv (UCI heart-disease dataset, 1025
rows but only 302 unique — exact duplicates are dropped before splitting).

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
    # The 1025-row CSV contains only 302 unique records (723 exact
    # duplicates). Without dropping them, copies of the same patient land in
    # both train and test, leaking labels and inflating the test metrics.
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {n_raw - len(df)} exact duplicate rows ({n_raw} -> {len(df)})")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # This Kaggle copy of UCI Cleveland has an INVERTED target: matched
    # row-by-row against UCI's processed.cleveland.data, target=0 is exactly
    # the 138 patients UCI diagnoses with disease (num > 0) and target=1 the
    # 164 without. The labels are left as-is (so the saved model's classes
    # stay [0, 1]); disease is reported as the positive class instead, and
    # prediction_service maps P(label 0) to elevated risk.
    run_training("heart", build_pipeline(), X, y, pos_label=0)


if __name__ == "__main__":
    main()
