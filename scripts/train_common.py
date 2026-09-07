"""
Shared training/evaluation helpers for the three train_*.py scripts
(Sections 3.7-3.10 of the methodology): stratified 80/20 split, 5-fold
cross-validation on the training fold, then a held-out test-set evaluation
with accuracy/precision/recall/F1 and a confusion matrix.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

MODELS_DIR = Path(__file__).resolve().parent.parent / "app" / "ml_models"


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def cross_validate_report(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5) -> dict:
    """5-fold stratified CV on the training fold only (test fold stays untouched)."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"],
        # n_jobs=1: loky's multiprocessing backend hits a cloudpickle
        # recursion error on this machine's Python 3.14 (too new for
        # loky/cloudpickle's process-spawning to reliably pickle closures
        # yet). These datasets are small enough that single-process CV is
        # still fast.
        n_jobs=1,
    )
    summary = {
        metric: {"mean": round(float(np.mean(scores[f"test_{metric}"])), 4), "std": round(float(np.std(scores[f"test_{metric}"])), 4)}
        for metric in ["accuracy", "precision", "recall", "f1"]
    }
    return summary


def evaluate_on_test_set(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    accuracy = float((y_pred == y_test.to_numpy()).mean())
    balanced_accuracy = float(balanced_accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    return {
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": {"labels": [0, 1], "matrix": cm},
    }


def print_correlation_report(X: pd.DataFrame, y: pd.Series, numeric_columns: list[str]) -> None:
    """Pearson correlation of each raw numeric feature with the target,
    before any preprocessing — a cheap first look for signal."""
    print("\n-- Raw correlation of numeric features with the target --")
    corr = X[numeric_columns].assign(_target=y.to_numpy()).corr()["_target"].drop("_target").sort_values(key=abs, ascending=False)
    for col, value in corr.items():
        print(f"  {col:25s} {value: .4f}")


def print_coefficient_report(pipeline: Pipeline, top_n: int = 15) -> float:
    """
    Prints the fitted LogisticRegression's coefficients against the
    post-preprocessing feature names, largest |magnitude| first, and returns
    the max absolute coefficient. This is the sanity check that caught
    Hypertension.csv having no real signal (every coefficient was ~0.001-0.01
    there, with the intercept alone explaining the base rate) — run it on
    every model before trusting its metrics, not just when something looks
    off.
    """
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_[0]
    order = np.argsort(-np.abs(coefs))

    print(f"\n-- Logistic regression coefficients (top {min(top_n, len(order))} by |magnitude|) --")
    for i in order[:top_n]:
        print(f"  {feature_names[i]:35s} {coefs[i]: .4f}")

    max_abs = float(np.max(np.abs(coefs)))
    note = "  <-- near zero: little/no learned signal" if max_abs < 0.05 else ""
    print(f"  max |coefficient| = {max_abs:.4f}{note}")
    return max_abs


def run_training(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
) -> Pipeline:
    """Splits, cross-validates, fits, evaluates, prints a report, saves the
    fitted pipeline + a metrics JSON, and returns the fitted pipeline."""
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"Rows: {len(X)}  |  Positive class rate: {y.mean():.4f}")

    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)} (80/20 stratified split)")

    majority_baseline = float(max(y.mean(), 1 - y.mean()))
    print(f"Majority-class baseline accuracy (always predict the more common label): {majority_baseline:.4f}")

    print(f"\n-- {cv_folds}-fold cross-validation on the training set --")
    cv_report = cross_validate_report(pipeline, X_train, y_train, cv_folds=cv_folds)
    for metric, stats in cv_report.items():
        print(f"  {metric:10s} mean={stats['mean']:.4f}  std={stats['std']:.4f}")

    pipeline.fit(X_train, y_train)
    print_coefficient_report(pipeline)

    print("\n-- Held-out test set evaluation --")
    test_report = evaluate_on_test_set(pipeline, X_test, y_test)
    print(f"  accuracy:          {test_report['accuracy']:.4f}  (majority-class baseline: {majority_baseline:.4f})")
    print(f"  balanced accuracy: {test_report['balanced_accuracy']:.4f}  (random-chance baseline: 0.5000)")
    print(f"  precision:         {test_report['precision']:.4f}")
    print(f"  recall:            {test_report['recall']:.4f}")
    print(f"  f1:                {test_report['f1']:.4f}")
    tn, fp, fn, tp = np.array(test_report["confusion_matrix"]["matrix"]).ravel()
    print("  confusion matrix (rows=actual, cols=predicted; labels=[0,1]):")
    print(f"    [[TN={tn:>6} FP={fp:>6}]")
    print(f"     [FN={fn:>6} TP={tp:>6}]]")

    # Raw accuracy vs. the majority-class baseline is misleading under class
    # imbalance (a model can score well below that baseline while still
    # having learned something real, if it trades accuracy for recall on the
    # minority class — see the diabetes model). Balanced accuracy — the
    # average of per-class recall — isn't fooled by imbalance: a classifier
    # that has learned nothing scores ~0.5 there regardless of class ratio.
    balanced_lift = test_report["balanced_accuracy"] - 0.5
    if balanced_lift < 0.02:
        print(
            f"\n  WARNING: balanced accuracy ({test_report['balanced_accuracy']:.4f}) is barely above "
            f"the random-chance baseline (0.5000), lift={balanced_lift:+.4f}. This means the model has "
            "found little to no learnable relationship between the features and the label in this "
            "dataset — treat its predictions with real caution."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = MODELS_DIR / f"{name}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {"majority_baseline_accuracy": round(majority_baseline, 4), "cross_validation": cv_report, "test_set": test_report},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved metrics to {metrics_path}")

    model_path = MODELS_DIR / f"{name}_model.pkl"
    import joblib

    joblib.dump(pipeline, model_path)
    print(f"Saved fitted pipeline to {model_path}")

    return pipeline
