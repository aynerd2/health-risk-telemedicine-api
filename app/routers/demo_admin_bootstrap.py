"""
############################################################################
# TEMPORARY / DEMO-ONLY — DELETE THIS FILE BEFORE GOING TO PRODUCTION.
#
# This exposes an unauthenticated way to create an admin account over HTTP,
# gated only by a shared-secret string. That's fine for a live demo where
# you need to bootstrap an admin without shelling into the server to run
# `python -m app.core.bootstrap`, and it's *not* the same hole as letting
# POST /api/v1/auth/register accept role="admin" (that endpoint deliberately
# rejects it — see app/routers/auth.py) — but a shared secret in an env var
# is still meaningfully weaker than "no route exists at all", so this
# should not survive past the demo.
#
# To remove it later:
#   1. Delete this file.
#   2. Remove the `app.include_router(demo_admin_bootstrap.router)` block
#      in app/main.py (it's commented and easy to find).
#   3. Remove DEMO_ADMIN_SECRET from .env / .env.example / config.py.
#
# In the meantime, the route only exists at all if DEMO_ADMIN_SECRET is set
# (see main.py) — leaving that env var unset in production is a one-line
# way to make sure this can't be hit even if step 1/2 above get missed.
############################################################################
"""
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, EmailStr
from sqlmodel import Session, select

from app.core.bootstrap import create_admin_user
from app.core.config import settings
from app.core.database import get_session
from app.models.entities import User, UserRole
from app.schemas.auth import UserRead

router = APIRouter(prefix="/api/v1/admin", tags=["Demo (temporary)"])


class _DemoAdminBootstrapRequest(BaseModel):
    """Deliberately its own schema (not RegisterRequest) — there's no `role`
    field here to avoid implying the caller can choose one; this endpoint
    only ever creates an admin."""

    full_name: str
    email: EmailStr
    password: str


def _check_demo_secret(secret_header: str | None, secret_query: str | None) -> None:
    # Fail closed: if the operator never set a secret, this endpoint is
    # unusable no matter what a caller sends (an empty/missing configured
    # secret must never match an empty/missing supplied one).
    configured = settings.DEMO_ADMIN_SECRET
    supplied = secret_header or secret_query
    if not configured or not supplied or not secrets.compare_digest(supplied, configured):
        # 404, not 401/403 — an unlisted demo route shouldn't even confirm
        # its own existence to someone probing without the secret.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")


@router.post("/bootstrap-demo", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def bootstrap_demo_admin(
    payload: _DemoAdminBootstrapRequest,
    session: Session = Depends(get_session),
    x_demo_admin_secret: str | None = Header(default=None),
    secret: str | None = Query(default=None),
):
    """
    TEMPORARY/DEMO-ONLY (see module docstring). Creates exactly one admin
    account and then refuses to create another — call it once per demo
    environment. Pass the secret either as a header:

        curl -X POST 'https://<host>/api/v1/admin/bootstrap-demo' \\
          -H 'X-Demo-Admin-Secret: <DEMO_ADMIN_SECRET>' \\
          -H 'Content-Type: application/json' \\
          -d '{"full_name":"Demo Admin","email":"admin@demo.local","password":"..."}'

    or as a query param (`?secret=...`) if that's easier to trigger from a
    browser during the demo.
    """
    _check_demo_secret(x_demo_admin_secret, secret)

    existing_admin = session.exec(select(User).where(User.role == UserRole.admin)).first()
    if existing_admin:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An admin account already exists — this demo route only ever creates one.",
        )

    try:
        user = create_admin_user(session, email=payload.email, password=payload.password, full_name=payload.full_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return UserRead(user_id=user.user_id, full_name=user.full_name, email=user.email, role=user.role)
