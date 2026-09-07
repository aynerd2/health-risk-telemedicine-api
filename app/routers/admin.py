"""
Administrator module (Section 3.2.6): doctor-approval workflow, user account
management, and basic platform usage reporting.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, func, select

from app.core.database import get_session
from app.core.deps import require_role
from app.models.entities import (
    Appointment,
    Consultation,
    Doctor,
    Patient,
    Prediction,
    RiskClass,
    User,
    UserRole,
)
from app.schemas.admin import PendingDoctorRead, PlatformStats, UserAdminRead

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["Administration"],
    dependencies=[Depends(require_role(UserRole.admin))],
)


@router.get("/users", response_model=list[UserAdminRead])
def list_users(session: Session = Depends(get_session)):
    return session.exec(select(User)).all()


@router.post("/users/{user_id}/deactivate", response_model=UserAdminRead)
def deactivate_user(user_id: int, session: Session = Depends(get_session)):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = False
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@router.post("/users/{user_id}/activate", response_model=UserAdminRead)
def activate_user(user_id: int, session: Session = Depends(get_session)):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = True
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@router.get("/doctors/pending", response_model=list[PendingDoctorRead])
def pending_doctors(session: Session = Depends(get_session)):
    rows = session.exec(
        select(Doctor, User).join(User, Doctor.user_id == User.user_id).where(Doctor.is_approved == False)  # noqa: E712
    ).all()
    return [
        PendingDoctorRead(
            doctor_id=doctor.doctor_id,
            user_id=user.user_id,
            full_name=user.full_name,
            email=user.email,
            specialty=doctor.specialty,
            license_no=doctor.license_no,
        )
        for doctor, user in rows
    ]


@router.post("/doctors/{doctor_id}/approve", response_model=PendingDoctorRead)
def approve_doctor(doctor_id: int, session: Session = Depends(get_session)):
    doctor = session.get(Doctor, doctor_id)
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    doctor.is_approved = True
    session.add(doctor)
    session.commit()
    session.refresh(doctor)

    user = session.get(User, doctor.user_id)
    return PendingDoctorRead(
        doctor_id=doctor.doctor_id,
        user_id=user.user_id,
        full_name=user.full_name,
        email=user.email,
        specialty=doctor.specialty,
        license_no=doctor.license_no,
    )


@router.get("/stats", response_model=PlatformStats)
def platform_stats(session: Session = Depends(get_session)):
    total_users = session.exec(select(func.count()).select_from(User)).one()
    total_patients = session.exec(select(func.count()).select_from(Patient)).one()
    total_doctors = session.exec(select(func.count()).select_from(Doctor)).one()
    pending_doctor_approvals = session.exec(
        select(func.count()).select_from(Doctor).where(Doctor.is_approved == False)  # noqa: E712
    ).one()
    total_predictions = session.exec(select(func.count()).select_from(Prediction)).one()
    elevated_predictions = session.exec(
        select(func.count()).select_from(Prediction).where(Prediction.risk_class == RiskClass.elevated)
    ).one()
    total_appointments = session.exec(select(func.count()).select_from(Appointment)).one()
    completed_consultations = session.exec(select(func.count()).select_from(Consultation)).one()

    return PlatformStats(
        total_users=total_users,
        total_patients=total_patients,
        total_doctors=total_doctors,
        pending_doctor_approvals=pending_doctor_approvals,
        total_predictions=total_predictions,
        elevated_predictions=elevated_predictions,
        total_appointments=total_appointments,
        completed_consultations=completed_consultations,
    )
