from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.entities import (
    ChestPainType,
    Condition,
    RestingECG,
    RiskClass,
    SexOption,
    SmokingHistory,
    STSlope,
    Thalassemia,
)


class HealthIntakeRequest(BaseModel):
    """
    Validated against the patient's health-intake form (Section 3.2.4) before
    it reaches the pre-processing pipeline described in Section 3.7.

    Only age and sex are required. Everything else is optional: the three
    trained models each pull the subset of these fields they were trained on
    (see app/services/prediction_service.py), and each model's own pipeline
    median/mode-imputes whatever the patient leaves blank — a real intake
    form shouldn't force someone to have a full lipid panel on hand before
    they can get a risk indication. A few fields feed more than one model
    under a different representation — e.g. `cholesterol`/`blood_glucose_mgdl`
    are raw mg/dL for the heart/diabetes models but get bucketed into the
    hypertension model's 3-level normal/above-normal/well-above-normal
    categories (see the _*_category() helpers in prediction_service.py).
    """

    # Demographics — used by all three models
    age: int = Field(ge=0, le=120)
    sex: SexOption

    # Vitals & labs
    systolic_bp: float | None = Field(default=None, ge=0, description="mmHg — heart-disease model only")
    cholesterol: float | None = Field(default=None, ge=0, description="Total cholesterol, mg/dL")
    blood_glucose_mgdl: float | None = Field(default=None, ge=0, description="mg/dL")
    hba1c_level: float | None = Field(default=None, ge=0, description="% — diabetes model only")
    bmi: float | None = Field(default=None, ge=0, description="diabetes + hypertension models")
    max_heart_rate: float | None = Field(default=None, ge=0, description="bpm during a stress/exercise test")

    # Cardiac history / exam findings — heart-disease model. These come from
    # a clinician's exam rather than patient self-report (fluoroscopy vessel
    # count, thalassemia type, ST-segment findings); a nurse/doctor would
    # typically fill these in during an initial visit.
    chest_pain_type: ChestPainType | None = None
    exercise_angina: bool | None = None
    resting_ecg: RestingECG | None = None
    st_slope: STSlope | None = None
    st_depression: float | None = Field(default=None, description="ST depression induced by exercise (oldpeak)")
    major_vessels_colored: int | None = Field(default=None, ge=0, le=4)
    thalassemia: Thalassemia | None = None

    # Lifestyle
    smoking_history: SmokingHistory | None = Field(default=None, description="diabetes + hypertension models")
    drinks_alcohol: bool | None = Field(default=None, description="hypertension model only")
    physically_active: bool | None = Field(default=None, description="hypertension model only")

    # Prior-diagnosis flags — the diabetes model uses these as inputs
    has_hypertension: bool | None = None
    has_heart_disease: bool | None = None


class ConditionRisk(BaseModel):
    condition: Condition
    risk_score: float  # probability, Equation 3.2
    risk_class: RiskClass


class PredictionResponse(BaseModel):
    record_id: int
    results: list[ConditionRisk]
    any_elevated: bool


class PredictionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    prediction_id: int
    condition: Condition
    risk_score: float
    risk_class: RiskClass
    date_generated: datetime


class HealthRecordHistory(BaseModel):
    record_id: int
    date_recorded: datetime
    predictions: list[PredictionRead]
