"""Password hashing/verification via bcrypt.

Use the `bcrypt` library directly; passlib's wrapper is incompatible
with bcrypt 4.x (looks up `bcrypt.__about__.__version__` which was
removed). Direct usage is also less abstraction.
"""

from __future__ import annotations

import bcrypt


def hash_password(plain: str) -> str:
    """Generate a bcrypt hash (cost factor 12). For setup/admin use."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time bcrypt verification. Returns False on any error."""
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False
