"""
Exercises the REAL trained .pkl pipelines (not the MagicMock stub every
other test uses — see conftest.py's stub_ml_models) to catch mistakes the
mocked tests can't: a field-name typo in a _*_frame() builder, a category
value that doesn't match training data, a dtype that breaks the
ColumnTransformer, etc. Skips itself if the models haven't been trained yet.
"""
import pytest

from app.schemas.prediction import HealthIntakeRequest
from app.services import prediction_service as ps

pytestmark = pytest.mark.usefixtures("client")  # spin up the app once (see module docstring)


def _reload_real_models():
    ps.load_models()
    if any(model is None for model in ps._MODELS.values()):
        pytest.skip("Trained .pkl files not present in app/ml_models — run scripts/train_*.py first")


FULL_INTAKE = HealthIntakeRequest(
    age=52,
    sex="male",
    systolic_bp=135,
    cholesterol=210,
    blood_glucose_mgdl=110,
    hba1c_level=5.9,
    bmi=28.4,
    max_heart_rate=145,
    chest_pain_type=1,
    exercise_angina=True,
    resting_ecg=0,
    st_slope=2,
    st_depression=1.2,
    major_vessels_colored=1,
    thalassemia=2,
    smoking_history="former",
    drinks_alcohol=False,
    physically_active=True,
    has_hypertension=False,
    has_heart_disease=False,
)

MINIMAL_INTAKE = HealthIntakeRequest(age=40, sex="female")

OTHER_SEX_INTAKE = HealthIntakeRequest(age=30, sex="other")


@pytest.mark.parametrize("intake", [FULL_INTAKE, MINIMAL_INTAKE, OTHER_SEX_INTAKE], ids=["full", "minimal", "other_sex"])
def test_real_models_predict_without_error(intake):
    _reload_real_models()
    results = ps.predict_all(intake)
    assert {r.condition.value for r in results} == {"heart_disease", "diabetes", "hypertension"}
    for r in results:
        assert 0.0 <= r.risk_score <= 1.0
