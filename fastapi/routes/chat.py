from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

router = APIRouter()

class Message(BaseModel):
    text: str

@router.post("/message")
async def create_message(message: Message):
    # TO DO: implement message creation
    return {"message": "Message created"}
