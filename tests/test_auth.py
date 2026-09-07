def register_patient(client, email="patient@example.com", password="secretpass123"):
    return client.post(
        "/api/v1/auth/register",
        json={"full_name": "Pat Ient", "email": email, "password": password, "role": "patient"},
    )


def test_register_and_login(client):
    resp = register_patient(client)
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == "patient@example.com"
    assert body["role"] == "patient"

    login = client.post(
        "/api/v1/auth/login", data={"username": "patient@example.com", "password": "secretpass123"}
    )
    assert login.status_code == 200
    assert login.json()["role"] == "patient"
    assert "access_token" in login.json()


def test_register_duplicate_email_rejected(client):
    register_patient(client)
    resp = register_patient(client)
    assert resp.status_code == 400


def test_first_admin_registration_allowed_when_none_exists(client):
    resp = client.post(
        "/api/v1/auth/register",
        json={"full_name": "First Admin", "email": "first-admin@example.com", "password": "secretpass123", "role": "admin"},
    )
    assert resp.status_code == 201
    assert resp.json()["role"] == "admin"


def test_second_admin_registration_rejected(client):
    first = client.post(
        "/api/v1/auth/register",
        json={"full_name": "First Admin", "email": "first-admin@example.com", "password": "secretpass123", "role": "admin"},
    )
    assert first.status_code == 201

    resp = client.post(
        "/api/v1/auth/register",
        json={"full_name": "Sneaky", "email": "sneaky@example.com", "password": "secretpass123", "role": "admin"},
    )
    assert resp.status_code == 400


def test_login_wrong_password_rejected(client):
    register_patient(client)
    resp = client.post("/api/v1/auth/login", data={"username": "patient@example.com", "password": "wrong"})
    assert resp.status_code == 401


def test_protected_endpoint_requires_token(client):
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 401


def test_me_returns_current_user(client):
    register_patient(client)
    login = client.post(
        "/api/v1/auth/login", data={"username": "patient@example.com", "password": "secretpass123"}
    ).json()
    resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {login['access_token']}"})
    assert resp.status_code == 200
    assert resp.json()["email"] == "patient@example.com"


def test_refresh_token_issues_new_access_token(client):
    register_patient(client)
    login = client.post(
        "/api/v1/auth/login", data={"username": "patient@example.com", "password": "secretpass123"}
    ).json()
    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": login["refresh_token"]})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_refresh_rejects_access_token(client):
    register_patient(client)
    login = client.post(
        "/api/v1/auth/login", data={"username": "patient@example.com", "password": "secretpass123"}
    ).json()
    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": login["access_token"]})
    assert resp.status_code == 401


def test_password_reset_flow(client):
    register_patient(client)
    forgot = client.post("/api/v1/auth/forgot-password", json={"email": "patient@example.com"})
    assert forgot.status_code == 200
    reset_token = forgot.json()["reset_token"]
    assert reset_token

    reset = client.post(
        "/api/v1/auth/reset-password", json={"reset_token": reset_token, "new_password": "newpassword456"}
    )
    assert reset.status_code == 200

    old_login = client.post(
        "/api/v1/auth/login", data={"username": "patient@example.com", "password": "secretpass123"}
    )
    assert old_login.status_code == 401

    new_login = client.post(
        "/api/v1/auth/login", data={"username": "patient@example.com", "password": "newpassword456"}
    )
    assert new_login.status_code == 200


def test_forgot_password_unknown_email_does_not_leak(client):
    resp = client.post("/api/v1/auth/forgot-password", json={"email": "nobody@example.com"})
    assert resp.status_code == 200
    assert resp.json()["reset_token"] == ""
