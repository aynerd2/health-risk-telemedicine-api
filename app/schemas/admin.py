from datetime import datetime

from pydantic import BaseModel

from app.models.entities import UserRole


class UserAdminRead(BaseModel):
    user_id: int
    full_name: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime


class PendingDoctorRead(BaseModel):
    doctor_id: int
    user_id: int
    full_name: str
    email: str
    specialty: str | None
    license_no: str | None


class PlatformStats(BaseModel):
    total_users: int
    total_patients: int
    total_doctors: int
    pending_doctor_approvals: int
    total_predictions: int
    elevated_predictions: int
    total_appointments: int
    completed_consultations: int
