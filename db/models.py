from pydantic import BaseModel


class User(BaseModel):
    user_id: str
    email: str
    subscription: str
    remaining_messages: int
