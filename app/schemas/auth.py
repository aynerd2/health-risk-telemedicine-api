from pydantic import BaseModel, EmailStr

from app.models.entities import UserRole


class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: UserRole = UserRole.patient


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    role: UserRole


class UserRead(BaseModel):
    user_id: int
    full_name: str
    email: EmailStr
    role: UserRole


class RefreshRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    # There is no email service wired up in this prototype, so the reset token
    # is returned directly in the response instead of being emailed. Swap this
    # for a "check your email" message once an email provider is configured.
    reset_token: str


class ResetPasswordRequest(BaseModel):
    reset_token: str
    new_password: str
