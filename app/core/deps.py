"""
Reusable FastAPI dependencies for authentication and role-based access
control (replaces Django's @login_required / permission classes).
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session, select

from app.core.database import get_session
from app.core.security import decode_token
from app.models.entities import Doctor, User, UserRole

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        raise credentials_exception

    user_id = payload.get("sub")
    user = session.get(User, int(user_id)) if user_id else None
    if not user or not user.is_active:
        raise credentials_exception
    return user


def require_role(*allowed_roles: UserRole):
    """Usage: Depends(require_role(UserRole.doctor))"""

    def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for this action")
        return user

    return checker


def require_approved_doctor(
    user: User = Depends(require_role(UserRole.doctor)),
    session: Session = Depends(get_session),
) -> Doctor:
    """Gate clinical actions (accepting appointments, writing consultation
    notes) behind admin approval, per the doctor-approval workflow."""
    doctor = session.exec(select(Doctor).where(Doctor.user_id == user.user_id)).first()
    if not doctor:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Doctor profile not found")
    if not doctor.is_approved:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your doctor account is pending admin approval",
        )
    return doctor
