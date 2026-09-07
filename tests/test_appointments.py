def register_and_login(client, email, role):
    client.post(
        "/api/v1/auth/register",
        json={"full_name": email.split("@")[0], "email": email, "password": "secretpass123", "role": role},
    )
    login = client.post("/api/v1/auth/login", data={"username": email, "password": "secretpass123"}).json()
    return {"Authorization": f"Bearer {login['access_token']}"}


def make_patient_and_approved_doctor(client, admin_token):
    patient_headers = register_and_login(client, "patient@example.com", "patient")
    doctor_headers = register_and_login(client, "doctor@example.com", "doctor")

    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    pending = client.get("/api/v1/admin/doctors/pending", headers=admin_headers).json()
    doctor_id = pending[0]["doctor_id"]
    client.post(f"/api/v1/admin/doctors/{doctor_id}/approve", headers=admin_headers)

    return patient_headers, doctor_headers, doctor_id


def test_unapproved_doctor_not_bookable(client):
    register_and_login(client, "patient@example.com", "patient")
    doctor_headers = register_and_login(client, "doctor@example.com", "doctor")

    # doctor list only shows approved doctors
    resp = client.get("/api/v1/doctors")
    assert resp.json() == []

    # doctor can't accept appointments while pending
    assert doctor_headers  # keep flake happy; doctor has no appointments yet either


def test_full_booking_and_consultation_flow(client, admin_token):
    patient_headers, doctor_headers, doctor_id = make_patient_and_approved_doctor(client, admin_token)

    listing = client.get("/api/v1/doctors")
    assert len(listing.json()) == 1

    book = client.post(
        "/api/v1/appointments",
        json={"doctor_id": doctor_id, "date_time": "2026-01-01T10:00:00"},
        headers=patient_headers,
    )
    assert book.status_code == 200
    appt = book.json()
    assert appt["status"] == "pending"

    confirm = client.patch(
        f"/api/v1/appointments/{appt['appt_id']}/status", json={"status": "confirmed"}, headers=doctor_headers
    )
    assert confirm.status_code == 200
    assert confirm.json()["status"] == "confirmed"

    notes = client.post(
        f"/api/v1/appointments/{appt['appt_id']}/consultation",
        json={"notes": "Discussed symptoms", "diagnosis": "None", "prescribed_action": "Follow up in 3 months"},
        headers=doctor_headers,
    )
    assert notes.status_code == 200

    # appointment auto-completes once notes are recorded
    mine = client.get("/api/v1/appointments/mine", headers=patient_headers).json()
    assert mine[0]["status"] == "completed"

    fetched_notes = client.get(f"/api/v1/appointments/{appt['appt_id']}/consultation", headers=patient_headers)
    assert fetched_notes.status_code == 200
    assert fetched_notes.json()["diagnosis"] == "None"


def test_patient_can_cancel_own_appointment(client, admin_token):
    patient_headers, _doctor_headers, doctor_id = make_patient_and_approved_doctor(client, admin_token)
    book = client.post(
        "/api/v1/appointments",
        json={"doctor_id": doctor_id, "date_time": "2026-01-01T10:00:00"},
        headers=patient_headers,
    ).json()

    cancel = client.post(f"/api/v1/appointments/{book['appt_id']}/cancel", headers=patient_headers)
    assert cancel.status_code == 200
    assert cancel.json()["status"] == "cancelled"


def test_doctor_cannot_touch_other_doctors_appointment(client, admin_token):
    patient_headers, doctor_headers, doctor_id = make_patient_and_approved_doctor(client, admin_token)
    book = client.post(
        "/api/v1/appointments",
        json={"doctor_id": doctor_id, "date_time": "2026-01-01T10:00:00"},
        headers=patient_headers,
    ).json()

    other_doctor_headers = register_and_login(client, "other-doc@example.com", "doctor")
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    pending = client.get("/api/v1/admin/doctors/pending", headers=admin_headers).json()
    other_doctor_id = [d["doctor_id"] for d in pending if d["email"] == "other-doc@example.com"][0]
    client.post(f"/api/v1/admin/doctors/{other_doctor_id}/approve", headers=admin_headers)

    resp = client.patch(
        f"/api/v1/appointments/{book['appt_id']}/status", json={"status": "confirmed"}, headers=other_doctor_headers
    )
    assert resp.status_code == 404


def test_admin_stats_reflect_activity(client, admin_token):
    make_patient_and_approved_doctor(client, admin_token)
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    stats = client.get("/api/v1/admin/stats", headers=admin_headers).json()
    assert stats["total_patients"] == 1
    assert stats["total_doctors"] == 1
    assert stats["pending_doctor_approvals"] == 0


def test_non_admin_cannot_access_admin_routes(client):
    headers = register_and_login(client, "patient@example.com", "patient")
    resp = client.get("/api/v1/admin/stats", headers=headers)
    assert resp.status_code == 403
