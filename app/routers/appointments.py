"""
Appointments & consultations (Sections 3.2.5/3.2.6): patients book a
teleconsultation with an approved doctor, the doctor confirms/cancels/
completes it, and consultation notes are recorded against the appointment.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.core.database import get_session
from app.core.deps import require_approved_doctor, require_role
from app.models.entities import (
    Appointment,
    AppointmentStatus,
    Consultation,
    Doctor,
    Patient,
    User,
    UserRole,
)
from app.schemas.appointment import (
    AppointmentRead,
    AppointmentStatusUpdate,
    BookAppointmentRequest,
    ConsultationCreate,
    ConsultationRead,
    DoctorRead,
)

router = APIRouter(prefix="/api/v1/appointments", tags=["Appointments"])
doctors_router = APIRouter(prefix="/api/v1/doctors", tags=["Appointments"])


@doctors_router.get("", response_model=list[DoctorRead])
def list_approved_doctors(session: Session = Depends(get_session)):
    """Patients pick from this list when booking (Section 3.2.5)."""
    rows = session.exec(
        select(Doctor, User)
        .join(User, Doctor.user_id == User.user_id)
        .where(Doctor.is_approved == True)  # noqa: E712
    ).all()
    return [
        DoctorRead(doctor_id=doctor.doctor_id, full_name=user.full_name, specialty=doctor.specialty, is_approved=True)
        for doctor, user in rows
    ]


@router.post("", response_model=AppointmentRead)
def book_appointment(
    payload: BookAppointmentRequest,
    current_user: User = Depends(require_role(UserRole.patient)),
    session: Session = Depends(get_session),
):
    patient = session.exec(select(Patient).where(Patient.user_id == current_user.user_id)).first()
    doctor = session.get(Doctor, payload.doctor_id)
    if not patient or not doctor:
        raise HTTPException(status_code=404, detail="Patient or doctor not found")
    if not doctor.is_approved:
        raise HTTPException(status_code=400, detail="This doctor is not yet approved to take appointments")

    appt = Appointment(patient_id=patient.patient_id, doctor_id=doctor.doctor_id, date_time=payload.date_time)
    session.add(appt)
    session.commit()
    session.refresh(appt)
    return appt


@router.get("/mine", response_model=list[AppointmentRead])
def my_appointments(
    current_user: User = Depends(require_role(UserRole.patient, UserRole.doctor)),
    session: Session = Depends(get_session),
):
    if current_user.role == UserRole.patient:
        patient = session.exec(select(Patient).where(Patient.user_id == current_user.user_id)).first()
        return session.exec(select(Appointment).where(Appointment.patient_id == patient.patient_id)).all()

    doctor = session.exec(select(Doctor).where(Doctor.user_id == current_user.user_id)).first()
    return session.exec(select(Appointment).where(Appointment.doctor_id == doctor.doctor_id)).all()


def _get_owned_appointment(appt_id: int, doctor: Doctor, session: Session) -> Appointment:
    appt = session.get(Appointment, appt_id)
    if not appt or appt.doctor_id != doctor.doctor_id:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return appt


@router.patch("/{appt_id}/status", response_model=AppointmentRead)
def update_appointment_status(
    appt_id: int,
    payload: AppointmentStatusUpdate,
    doctor: Doctor = Depends(require_approved_doctor),
    session: Session = Depends(get_session),
):
    """Doctor confirms, cancels, or marks an appointment completed."""
    appt = _get_owned_appointment(appt_id, doctor, session)
    appt.status = payload.status
    session.add(appt)
    session.commit()
    session.refresh(appt)
    return appt


@router.post("/{appt_id}/cancel", response_model=AppointmentRead)
def cancel_appointment(
    appt_id: int,
    current_user: User = Depends(require_role(UserRole.patient)),
    session: Session = Depends(get_session),
):
    """A patient may cancel their own pending/confirmed appointment."""
    patient = session.exec(select(Patient).where(Patient.user_id == current_user.user_id)).first()
    appt = session.get(Appointment, appt_id)
    if not appt or not patient or appt.patient_id != patient.patient_id:
        raise HTTPException(status_code=404, detail="Appointment not found")
    if appt.status == AppointmentStatus.completed:
        raise HTTPException(status_code=400, detail="Cannot cancel a completed appointment")

    appt.status = AppointmentStatus.cancelled
    session.add(appt)
    session.commit()
    session.refresh(appt)
    return appt


@router.post("/{appt_id}/consultation", response_model=ConsultationRead)
def record_consultation(
    appt_id: int,
    payload: ConsultationCreate,
    doctor: Doctor = Depends(require_approved_doctor),
    session: Session = Depends(get_session),
):
    """Doctor records notes/diagnosis at the end of a teleconsultation and
    marks the appointment completed."""
    appt = _get_owned_appointment(appt_id, doctor, session)

    existing = session.exec(select(Consultation).where(Consultation.appt_id == appt_id)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Consultation notes already recorded for this appointment")

    consultation = Consultation(appt_id=appt_id, **payload.model_dump())
    session.add(consultation)
    appt.status = AppointmentStatus.completed
    session.add(appt)
    session.commit()
    session.refresh(consultation)
    return consultation


@router.get("/{appt_id}/consultation", response_model=ConsultationRead)
def get_consultation(
    appt_id: int,
    current_user: User = Depends(require_role(UserRole.patient, UserRole.doctor)),
    session: Session = Depends(get_session),
):
    appt = session.get(Appointment, appt_id)
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")

    if current_user.role == UserRole.patient:
        patient = session.exec(select(Patient).where(Patient.user_id == current_user.user_id)).first()
        if not patient or appt.patient_id != patient.patient_id:
            raise HTTPException(status_code=404, detail="Appointment not found")
    else:
        doctor = session.exec(select(Doctor).where(Doctor.user_id == current_user.user_id)).first()
        if not doctor or appt.doctor_id != doctor.doctor_id:
            raise HTTPException(status_code=404, detail="Appointment not found")

    consultation = session.exec(select(Consultation).where(Consultation.appt_id == appt_id)).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="No consultation notes recorded yet")
    return consultation
