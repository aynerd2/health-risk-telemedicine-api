"""
One-off admin account creation. Admins can't self-register through the public
API (see app/routers/auth.py), so the very first admin has to be created
directly against the database:

    python -m app.core.bootstrap admin@example.com "a strong password" "Admin Name"
"""
import sys

from sqlmodel import Session, select

from app.core.database import engine, init_db
from app.core.security import hash_password
from app.models.entities import User, UserRole


def create_admin_user(session: Session, email: str, password: str, full_name: str) -> User:
    existing = session.exec(select(User).where(User.email == email)).first()
    if existing:
        raise ValueError(f"A user with email {email} already exists")

    user = User(full_name=full_name, email=email, password_hash=hash_password(password), role=UserRole.admin)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage: python -m app.core.bootstrap <email> <password> "<full name>"')
        raise SystemExit(1)

    init_db()
    with Session(engine) as session:
        admin = create_admin_user(session, email=sys.argv[1], password=sys.argv[2], full_name=sys.argv[3])
        print(f"Created admin user #{admin.user_id} ({admin.email})")
