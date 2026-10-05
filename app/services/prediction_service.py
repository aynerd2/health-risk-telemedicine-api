"""
AI Health Risk Prediction Module (Section 3.2.7).

Loads the three trained Logistic Regression pipelines once at application
start-up (see app/main.py's lifespan handler) and exposes a single
`predict_all` function that the prediction router calls. Each pipeline was
trained by backend/scripts/train_*.py against the datasets in
backend/training_data/ (Sections 3.7-3.10) and is a single sklearn Pipeline
combining a ColumnTransformer (impute + scale numeric columns, impute +
one-hot categorical columns) with LogisticRegression, so it accepts a
single-row pandas DataFrame with the exact column names it was trained on
and exposes predict_proba().

Hypertension model note: this was originally trained on Hypertension.csv,
whose label turned out to have no measurable relationship with any of its
feature columns (every correlation and logistic-regression coefficient was
~0, balanced accuracy exactly 0.5 — random chance). It's now trained instead
on the sulianova/cardiovascular-disease-dataset (cardio_train.csv): rather
than trust a pre-made label, the "hypertension" target is derived directly
from that dataset's own systolic/diastolic BP readings using the ACC/AHA
Stage-1 threshold (ap_hi >= 130 OR ap_lo >= 80), and the model is trained on
the *other* risk factors (age, BMI, cholesterol, glucose, smoking, alcohol,
activity, gender) — not on the BP readings themselves, since those would
trivially determine the label by definition. Balanced accuracy is 0.624
(vs. 0.5 chance) — a real but modest effect, which is medically expected:
these lifestyle/demographic factors are associated with hypertension risk
but don't fully determine an individual's blood pressure. See
app/ml_models/hypertension_metrics.json and scripts/train_hypertension_model.py.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings
from app.models.entities import ChestPainType, Condition, RestingECG, RiskClass, SexOption, SmokingHistory, STSlope, Thalassemia
from app.schemas.prediction import ConditionRisk, HealthIntakeRequest

DECISION_THRESHOLD = 0.5  # Section 3.9

# Which training label means "has the condition" for each model. The heart
# CSV (Kaggle johnsmith88 copy of UCI Cleveland) has its target INVERTED
# relative to UCI: matching all 302 unique rows back to UCI's
# processed.cleveland.data shows target=1 on exactly the 164 rows UCI
# diagnoses as no disease (num == 0) and target=0 on the 138 with disease.
# So for heart disease, P(elevated risk) is the probability of label 0.
_POSITIVE_LABEL = {
    Condition.heart_disease: 0,
    Condition.diabetes: 1,
    Condition.hypertension: 1,
}

_MODELS: dict[Condition, object] = {}

_MODEL_FILENAMES = {
    Condition.heart_disease: "heart_model.pkl",
    Condition.diabetes: "diabetes_model.pkl",
    Condition.hypertension: "hypertension_model.pkl",
}


def load_models() -> None:
    """Called once from the FastAPI startup event."""
    models_dir = Path(settings.ML_MODELS_DIR)
    for condition, filename in _MODEL_FILENAMES.items():
        path = models_dir / filename
        if path.exists():
            _MODELS[condition] = joblib.load(path)
        else:
            # Allows the API to boot (e.g. in CI, or before models are trained)
            # without crashing; predict_all will raise a clear error instead.
            _MODELS[condition] = None


# --- value mapping helpers -------------------------------------------------
# Each trained pipeline's OneHotEncoder(handle_unknown="ignore") only
# recognizes the exact category values/spellings present in its training
# CSV. Any value it hasn't seen is silently treated as "unknown" (encoded as
# all-zero), not an error — so getting these mappings exactly right matters.


def _bool_to_int(value: bool | None) -> float | None:
    if value is None:
        return None
    return 1.0 if value else 0.0


def _sex_to_heart_code(sex: SexOption) -> int:
    # The UCI heart-disease training data only encodes a binary sex column
    # (1=male, 0=female); "other" has no representation there, so it falls
    # back to the dataset's own encoding for female — a known limitation of
    # that dataset, not something this app can fix without retraining on
    # data that includes a third category.
    return 1 if sex == SexOption.male else 0


def _sex_to_label(sex: SexOption) -> str:
    # Diabetes.csv's gender column has exactly these three values.
    return {SexOption.male: "Male", SexOption.female: "Female", SexOption.other: "Other"}[sex]


def _sex_to_cardio_gender(sex: SexOption) -> int:
    # cardio_train.csv only encodes a binary gender column (1=female,
    # 2=male — confirmed by comparing mean height/weight per code); "other"
    # has no representation there, so it falls back to the dataset's female
    # code, the same limitation as the heart-disease dataset above.
    return 2 if sex == SexOption.male else 1


# The Kaggle heart CSV also renumbered cp/restecg/slope/thal relative to UCI
# (verified row-by-row against processed.cleveland.data), so the API's
# clinical enums can't be passed through as-is — these translate each
# clinical meaning to the code the heart model was actually trained on.
_HEART_CP_CODE = {
    ChestPainType.typical_angina: 3,
    ChestPainType.atypical_angina: 1,
    ChestPainType.non_anginal: 2,
    ChestPainType.asymptomatic: 0,
}
_HEART_RESTECG_CODE = {
    RestingECG.normal: 1,
    RestingECG.st_t_abnormality: 2,
    RestingECG.left_ventricular_hypertrophy: 0,
}
_HEART_SLOPE_CODE = {
    STSlope.upsloping: 2,
    STSlope.flat: 1,
    STSlope.downsloping: 0,
}
_HEART_THAL_CODE = {
    Thalassemia.unknown: 0,  # missing ('?') in UCI
    Thalassemia.normal: 2,
    Thalassemia.fixed_defect: 1,
    Thalassemia.reversible_defect: 3,
}


def _fbs_flag(blood_glucose_mgdl: float | None) -> float | None:
    """Heart model's `fbs` feature is "fasting blood sugar > 120 mg/dl" as a
    0/1 flag; the intake form collects a single glucose reading instead."""
    if blood_glucose_mgdl is None:
        return None
    return 1.0 if blood_glucose_mgdl > 120 else 0.0


def _cholesterol_category(mg_dl: float | None) -> int | None:
    """Buckets a raw total-cholesterol reading (mg/dL) into the 3-level
    category (1=normal, 2=above normal, 3=well above normal) the cardio
    dataset's hypertension model was trained on, using the standard NCEP
    ATP III cutoffs (normal <200, borderline-high 200-239, high >=240)."""
    if mg_dl is None:
        return None
    if mg_dl < 200:
        return 1
    if mg_dl < 240:
        return 2
    return 3


def _glucose_category(mg_dl: float | None) -> int | None:
    """Buckets a raw blood-glucose reading (mg/dL) into the 3-level category
    the hypertension model was trained on, using standard fasting-glucose
    cutoffs (normal <100, prediabetic 100-125, diabetic-range >=126)."""
    if mg_dl is None:
        return None
    if mg_dl < 100:
        return 1
    if mg_dl < 126:
        return 2
    return 3


def _smoker_flag(smoking_history: SmokingHistory | None) -> float | None:
    """cardio_train.csv's `smoke` is a simple current-smoker 0/1 flag,
    coarser than the diabetes model's 6-value smoking_history. "No Info"
    and "ever" (smoked at some point) don't say whether the patient smokes
    *now*, so they're treated as missing and imputed rather than guessed."""
    if smoking_history in (None, SmokingHistory.no_info, SmokingHistory.ever):
        return None
    return 1.0 if smoking_history == SmokingHistory.current else 0.0


# --- per-condition feature frames ------------------------------------------
# Column names/order below must match backend/scripts/train_*.py exactly.
# Missing (None) values must reach the pipeline as NaN — see build_frame()
# — so each pipeline's SimpleImputer fills them with the training set's
# median (numeric) or most frequent value (categorical), per Section 3.7.


def _heart_frame(intake: HealthIntakeRequest) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": intake.age,
                "trestbps": intake.systolic_bp,
                "chol": intake.cholesterol,
                "thalach": intake.max_heart_rate,
                "oldpeak": intake.st_depression,
                "sex": _sex_to_heart_code(intake.sex),
                "cp": _HEART_CP_CODE.get(intake.chest_pain_type),
                "fbs": _fbs_flag(intake.blood_glucose_mgdl),
                "restecg": _HEART_RESTECG_CODE.get(intake.resting_ecg),
                "exang": _bool_to_int(intake.exercise_angina),
                "slope": _HEART_SLOPE_CODE.get(intake.st_slope),
                "ca": intake.major_vessels_colored,
                "thal": _HEART_THAL_CODE.get(intake.thalassemia),
            }
        ]
    )


def _diabetes_frame(intake: HealthIntakeRequest) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": intake.age,
                "bmi": intake.bmi,
                "HbA1c_level": intake.hba1c_level,
                "blood_glucose_level": intake.blood_glucose_mgdl,
                "gender": _sex_to_label(intake.sex),
                "hypertension": _bool_to_int(intake.has_hypertension),
                "heart_disease": _bool_to_int(intake.has_heart_disease),
                "smoking_history": intake.smoking_history.value if intake.smoking_history is not None else None,
            }
        ]
    )


def _hypertension_frame(intake: HealthIntakeRequest) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": intake.age,
                "bmi": intake.bmi,
                "gender": _sex_to_cardio_gender(intake.sex),
                "cholesterol": _cholesterol_category(intake.cholesterol),
                "gluc": _glucose_category(intake.blood_glucose_mgdl),
                "smoke": _smoker_flag(intake.smoking_history),
                "alco": _bool_to_int(intake.drinks_alcohol),
                "active": _bool_to_int(intake.physically_active),
            }
        ]
    )


def build_frame(condition: Condition, intake: HealthIntakeRequest) -> pd.DataFrame:
    """The model-ready single-row frame for one condition. A blank field is
    None in an object-dtype column, and SimpleImputer only recognises NaN as
    missing there — None slipped past it unimputed and the OneHotEncoder
    then zeroed the whole feature as an "unknown" category. Converting to
    NaN makes blanks get the training-set mode/median as intended."""
    frame = _FRAME_BUILDERS[condition](intake)
    return frame.where(frame.notna(), np.nan)


_FRAME_BUILDERS = {
    Condition.heart_disease: _heart_frame,
    Condition.diabetes: _diabetes_frame,
    Condition.hypertension: _hypertension_frame,
}


def predict_one(condition: Condition, intake: HealthIntakeRequest) -> ConditionRisk:
    model = _MODELS.get(condition)
    if model is None:
        raise RuntimeError(
            f"Model for '{condition.value}' is not loaded. "
            f"Place the trained .pkl file in {settings.ML_MODELS_DIR}."
        )
    frame = build_frame(condition, intake)
    positive_column = list(model.classes_).index(_POSITIVE_LABEL[condition])
    probability = float(model.predict_proba(frame)[0][positive_column])  # P(elevated risk), Eq. 3.2
    risk_class = RiskClass.elevated if probability >= DECISION_THRESHOLD else RiskClass.low
    return ConditionRisk(condition=condition, risk_score=round(probability, 4), risk_class=risk_class)


def predict_all(intake: HealthIntakeRequest) -> list[ConditionRisk]:
    return [predict_one(condition, intake) for condition in Condition]
