from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select

from app.core.database import get_session
from app.core.deps import get_current_user
from app.core.security import (
    create_access_token,
    create_password_reset_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.models.entities import Doctor, Patient, User, UserRole
from app.schemas.auth import (
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    RefreshRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserRead,
)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, session: Session = Depends(get_session)):
    if payload.role == UserRole.admin:
        # TEMPORARY — REMOVE AFTER THE FIRST ADMIN ACCOUNT IS CREATED.
        # Public self-registration as admin is allowed *only* as a one-time
        # bootstrap when no admin exists yet at all (for environments like a
        # managed host where there's no shell access to run
        # `python -m app.core.bootstrap`). The moment one admin exists, this
        # closes itself automatically — every subsequent attempt hits the
        # `existing_admin` check below and is rejected exactly like the old
        # unconditional block was. Revert to unconditionally rejecting
        # role="admin" once you no longer need this bootstrap path, and
        # remove the matching temporary "Admin" option in the frontend's
        # register page.
        existing_admin = session.exec(select(User).where(User.role == UserRole.admin)).first()
        if existing_admin:
            raise HTTPException(status_code=400, detail="Cannot self-register as admin")

    existing = session.exec(select(User).where(User.email == payload.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        full_name=payload.full_name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        role=payload.role,
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    # Create the linked Patient/Doctor profile row (Section 3.5 ERD: USER 1:1 PATIENT/DOCTOR)
    if user.role == UserRole.patient:
        session.add(Patient(user_id=user.user_id))
    elif user.role == UserRole.doctor:
        session.add(Doctor(user_id=user.user_id))  # is_approved defaults to False (Section 3.2.6)
    session.commit()

    return UserRead(user_id=user.user_id, full_name=user.full_name, email=user.email, role=user.role)


@router.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive")

    return TokenResponse(
        access_token=create_access_token(str(user.user_id), user.role.value),
        refresh_token=create_refresh_token(str(user.user_id)),
        role=user.role,
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest, session: Session = Depends(get_session)):
    """Exchange a still-valid refresh token for a new access + refresh token pair."""
    data = decode_token(payload.refresh_token)
    if not data or data.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token")

    user = session.get(User, int(data["sub"]))
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token")

    return TokenResponse(
        access_token=create_access_token(str(user.user_id), user.role.value),
        refresh_token=create_refresh_token(str(user.user_id)),
        role=user.role,
    )


@router.get("/me", response_model=UserRead)
def me(current_user: User = Depends(get_current_user)):
    return UserRead(
        user_id=current_user.user_id,
        full_name=current_user.full_name,
        email=current_user.email,
        role=current_user.role,
    )


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
def forgot_password(payload: ForgotPasswordRequest, session: Session = Depends(get_session)):
    """
    Issues a short-lived password-reset token. No email provider is wired up
    in this prototype, so the token is returned directly in the response
    rather than delivered out-of-band — replace with an emailed link before
    shipping to real users. Always returns 200 (even for unknown emails) so
    this endpoint can't be used to enumerate registered accounts.
    """
    user = session.exec(select(User).where(User.email == payload.email)).first()
    if not user:
        return ForgotPasswordResponse(reset_token="")
    return ForgotPasswordResponse(reset_token=create_password_reset_token(str(user.user_id)))


@router.post("/reset-password", response_model=UserRead)
def reset_password(payload: ResetPasswordRequest, session: Session = Depends(get_session)):
    data = decode_token(payload.reset_token)
    if not data or data.get("type") != "reset":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired reset token")

    user = session.get(User, int(data["sub"]))
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired reset token")

    user.password_hash = hash_password(payload.new_password)
    session.add(user)
    session.commit()
    return UserRead(user_id=user.user_id, full_name=user.full_name, email=user.email, role=user.role)
