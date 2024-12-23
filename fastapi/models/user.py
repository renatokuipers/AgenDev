from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    id: int
    username: str
    email: str
    password: str
    role: str

    class Config:
        orm_mode = True
