"""
Train the hypertension risk model (Sections 3.7-3.10) on
backend/training_data/cardio_train.csv (sulianova/cardiovascular-disease-
dataset on Kaggle, ~70,000 rows).

The dataset originally used for this model (Hypertension.csv) had a label
with no measurable relationship to any of its columns — see the git history
of this file / BUILD_PLAN.md for the correlation and coefficient analysis
that caught it. This dataset avoids that failure mode by construction: we
derive the label ourselves from real systolic/diastolic blood pressure
readings using the ACC/AHA Stage-1-hypertension threshold
(ap_hi >= 130 OR ap_lo >= 80), rather than trusting a pre-made column, so
the label is guaranteed to be a genuine function of measured data.

Consequently the model is trained on the *remaining* risk factors — age,
BMI (computed from height/weight), cholesterol, glucose, smoking, alcohol,
physical activity, gender — and NOT on ap_hi/ap_lo themselves, since those
trivially determine the label by definition and training on them would just
be learning the threshold rule back, not testing whether the other risk
factors carry any signal.

Data cleaning: this dataset is known to contain physiologically implausible
rows (e.g. ap_hi/ap_lo negative or in the thousands, diastolic > systolic,
extreme height/weight) - Section 3.7 "handle missing/invalid values". Rows
outside plausible clinical ranges are dropped before the label is derived,
since a garbage BP reading would corrupt the label itself.

    python scripts/train_hypertension_model.py
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_common import print_correlation_report, run_training

DATA_PATH = Path(__file__).resolve().parent.parent / "training_data" / "cardio_train.csv"

NUMERIC_FEATURES = ["age", "bmi"]
CATEGORICAL_FEATURES = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
SYSTOLIC_THRESHOLD = 130
DIASTOLIC_THRESHOLD = 80


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")

    # Plausible clinical ranges — drops the known bad rows in this dataset
    # (e.g. ap_hi of -150 or 16020, ap_lo > ap_hi) before they can corrupt
    # the derived label or the BMI/age features.
    df = df[df["ap_hi"].between(70, 250) & df["ap_lo"].between(40, 150) & (df["ap_hi"] > df["ap_lo"])]
    df = df[df["height"].between(120, 220) & df["weight"].between(30, 200)]

    df["age"] = df["age"] / 365.25  # raw column is age in days
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2

    df["hypertension"] = ((df["ap_hi"] >= SYSTOLIC_THRESHOLD) | (df["ap_lo"] >= DIASTOLIC_THRESHOLD)).astype(int)

    return df


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
    # class_weight="balanced": the derived label is imbalanced (~82%
    # positive at the 130/80 threshold), so an unweighted model just
    # predicts "hypertensive" for almost everyone and balanced accuracy
    # barely beats chance even though the coefficients show real signal.
    return Pipeline([("preprocess", preprocessor), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def main() -> None:
    df = load_and_clean()
    print(f"Rows after cleaning implausible BP/height/weight readings: {len(df)} (from {len(pd.read_csv(DATA_PATH, sep=';'))})")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["hypertension"]

    print_correlation_report(X, y, NUMERIC_FEATURES)

    run_training("hypertension", build_pipeline(), X, y)


if __name__ == "__main__":
    main()
