from datetime import datetime

from pydantic import BaseModel

from app.models.entities import AppointmentStatus


class BookAppointmentRequest(BaseModel):
    doctor_id: int
    date_time: datetime


class AppointmentStatusUpdate(BaseModel):
    status: AppointmentStatus


class AppointmentRead(BaseModel):
    appt_id: int
    patient_id: int
    doctor_id: int
    date_time: datetime
    status: AppointmentStatus


class DoctorRead(BaseModel):
    doctor_id: int
    full_name: str
    specialty: str | None
    is_approved: bool


class ConsultationCreate(BaseModel):
    notes: str | None = None
    diagnosis: str | None = None
    prescribed_action: str | None = None


class ConsultationRead(BaseModel):
    consult_id: int
    appt_id: int
    notes: str | None
    diagnosis: str | None
    prescribed_action: str | None
    recorded_at: datetime
