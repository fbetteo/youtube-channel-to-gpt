from db.test_ps import connection, cache
from db.models import User
import passlib.hash as _hash


async def get_user_by_email(email: str):
    c = connection.cursor()
    c.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = c.fetchone()
    c.close()
    if user:
        return {"user_id": user[0], "user_name": user[1], "email": user[2]}
    else:
        return None


async def create_user(user: User):
    hashed_password = _hash.bcrypt.hash(user.password)
    c = connection.cursor()
    c.execute(
        "INSERT INTO users(email, hashed_password) VALUES (%s, %s)",
        (user.email, hashed_password),
    )
    c.close()
    return {"email": user.email, "pass": hashed_password}
