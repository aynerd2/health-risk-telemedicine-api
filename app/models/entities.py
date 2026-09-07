"""
SQLModel entities mirroring the Entity-Relationship Diagram in Figure 3.3:
USER -> PATIENT / DOCTOR, HEALTH_RECORD -> PREDICTION, APPOINTMENT -> CONSULTATION.
"""
from datetime import datetime
from enum import Enum

from sqlmodel import Field, SQLModel


class UserRole(str, Enum):
    patient = "patient"
    doctor = "doctor"
    admin = "admin"


class User(SQLModel, table=True):
    user_id: int | None = Field(default=None, primary_key=True)
    full_name: str
    email: str = Field(unique=True, index=True)
    password_hash: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Patient(SQLModel, table=True):
    patient_id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.user_id", unique=True)
    date_of_birth: datetime | None = None
    gender: str | None = None
    phone_no: str | None = None


class Doctor(SQLModel, table=True):
    doctor_id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.user_id", unique=True)
    specialty: str | None = None
    license_no: str | None = None
    is_approved: bool = False


class SexOption(str, Enum):
    male = "male"
    female = "female"
    other = "other"


class ChestPainType(int, Enum):
    """Matches the `cp` codes in the heart-disease training data."""

    typical_angina = 0
    atypical_angina = 1
    non_anginal = 2
    asymptomatic = 3


class RestingECG(int, Enum):
    """Matches the `restecg` codes in the heart-disease training data."""

    normal = 0
    st_t_abnormality = 1
    left_ventricular_hypertrophy = 2


class STSlope(int, Enum):
    """Matches the `slope` codes in the heart-disease training data."""

    upsloping = 0
    flat = 1
    downsloping = 2


class Thalassemia(int, Enum):
    """Matches the `thal` codes in the heart-disease training data."""

    unknown = 0
    normal = 1
    fixed_defect = 2
    reversible_defect = 3


class SmokingHistory(str, Enum):
    """Matches the `smoking_history` values in the diabetes training data."""

    never = "never"
    no_info = "No Info"
    current = "current"
    former = "former"
    ever = "ever"
    not_current = "not current"


class HealthRecord(SQLModel, table=True):
    """
    Clinical attributes (Table 3.2), covering every feature the three
    trained models actually consume (see app/services/prediction_service.py).
    Only age and sex are required — everything else is optional so the
    shared intake form doesn't force a patient to have a full lab panel on
    hand; each model's pipeline median/mode-imputes whatever is missing at
    prediction time (Section 3.7).
    """

    record_id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.patient_id", index=True)

    # Demographics — used by all three models
    age: int
    sex: SexOption

    # Vitals & labs
    systolic_bp: float | None = None  # heart model only (trestbps)
    cholesterol: float | None = None  # mg/dL; heart uses raw, hypertension buckets it
    blood_glucose_mgdl: float | None = None  # mg/dL; diabetes uses raw, heart/hypertension bucket it
    hba1c_level: float | None = None  # % — diabetes only
    bmi: float | None = None  # diabetes + hypertension
    max_heart_rate: float | None = None  # thalach: heart rate during a stress test

    # Cardiac history / exam findings — heart-disease model
    chest_pain_type: ChestPainType | None = None
    exercise_angina: bool | None = None
    resting_ecg: RestingECG | None = None
    st_slope: STSlope | None = None
    st_depression: float | None = None
    major_vessels_colored: int | None = None
    thalassemia: Thalassemia | None = None

    # Lifestyle
    smoking_history: SmokingHistory | None = None  # diabetes + hypertension (bucketed)
    drinks_alcohol: bool | None = None  # hypertension only
    physically_active: bool | None = None  # hypertension only

    # Prior-diagnosis flags — the diabetes model uses these as inputs
    has_hypertension: bool | None = None
    has_heart_disease: bool | None = None

    date_recorded: datetime = Field(default_factory=datetime.utcnow)


class RiskClass(str, Enum):
    low = "low"
    elevated = "elevated"


class Condition(str, Enum):
    heart_disease = "heart_disease"
    diabetes = "diabetes"
    hypertension = "hypertension"


class Prediction(SQLModel, table=True):
    prediction_id: int | None = Field(default=None, primary_key=True)
    record_id: int = Field(foreign_key="healthrecord.record_id", index=True)
    condition: Condition
    risk_score: float  # probability from the sigmoid output, Equation 3.2
    risk_class: RiskClass
    date_generated: datetime = Field(default_factory=datetime.utcnow)


class AppointmentStatus(str, Enum):
    pending = "pending"
    confirmed = "confirmed"
    completed = "completed"
    cancelled = "cancelled"


class Appointment(SQLModel, table=True):
    appt_id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.patient_id", index=True)
    doctor_id: int = Field(foreign_key="doctor.doctor_id", index=True)
    date_time: datetime
    status: AppointmentStatus = AppointmentStatus.pending


class Consultation(SQLModel, table=True):
    consult_id: int | None = Field(default=None, primary_key=True)
    appt_id: int = Field(foreign_key="appointment.appt_id", unique=True)
    notes: str | None = None
    diagnosis: str | None = None
    prescribed_action: str | None = None
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
