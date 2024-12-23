from pydantic import BaseModel

class Provider(BaseModel):
    name: str
    api_key: str
    temperature: float
    max_tokens: int
    top_p: float
    context_length: int
