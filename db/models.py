from pydantic import BaseModel


class User(BaseModel):
    email: str
    subscription: str
