VALID_INTAKE = {
    "age": 45,
    "sex": "male",
    "systolic_bp": 130,
    "cholesterol": 220,
    "blood_glucose_mgdl": 100,
    "hba1c_level": 5.6,
    "max_heart_rate": 150,
    "bmi": 27.5,
    "chest_pain_type": 0,
    "exercise_angina": False,
    "resting_ecg": 0,
    "st_slope": 1,
    "st_depression": 0.8,
    "major_vessels_colored": 0,
    "thalassemia": 2,
    "smoking_history": "never",
    "drinks_alcohol": False,
    "physically_active": True,
    "has_hypertension": False,
    "has_heart_disease": False,
}


def register_and_login(client, email="patient@example.com", role="patient"):
    client.post(
        "/api/v1/auth/register",
        json={"full_name": "Pat Ient", "email": email, "password": "secretpass123", "role": role},
    )
    login = client.post("/api/v1/auth/login", data={"username": email, "password": "secretpass123"}).json()
    return {"Authorization": f"Bearer {login['access_token']}"}


def test_submit_intake_returns_three_conditions(client):
    headers = register_and_login(client)
    resp = client.post("/api/v1/predictions", json=VALID_INTAKE, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    conditions = {r["condition"] for r in body["results"]}
    assert conditions == {"heart_disease", "diabetes", "hypertension"}
    # stub model always returns P(elevated) = 0.8
    assert body["any_elevated"] is True
    assert all(r["risk_class"] == "elevated" for r in body["results"])


def test_minimal_intake_only_age_and_sex(client):
    """Every other field is optional — each model's own SimpleImputer should
    fill in the rest."""
    headers = register_and_login(client)
    resp = client.post("/api/v1/predictions", json={"age": 50, "sex": "female"}, headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 3


def test_intake_rejects_invalid_age(client):
    headers = register_and_login(client)
    bad = {**VALID_INTAKE, "age": -5}
    resp = client.post("/api/v1/predictions", json=bad, headers=headers)
    assert resp.status_code == 422


def test_intake_requires_patient_role(client):
    headers = register_and_login(client, email="doc@example.com", role="doctor")
    resp = client.post("/api/v1/predictions", json=VALID_INTAKE, headers=headers)
    assert resp.status_code == 403


def test_prediction_history_lists_past_submissions(client):
    headers = register_and_login(client)
    client.post("/api/v1/predictions", json=VALID_INTAKE, headers=headers)
    client.post("/api/v1/predictions", json=VALID_INTAKE, headers=headers)

    resp = client.get("/api/v1/predictions/mine", headers=headers)
    assert resp.status_code == 200
    history = resp.json()
    assert len(history) == 2
    assert len(history[0]["predictions"]) == 3


def test_unauthenticated_intake_rejected(client):
    resp = client.post("/api/v1/predictions", json=VALID_INTAKE)
    assert resp.status_code == 401
