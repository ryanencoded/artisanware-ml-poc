from pydantic import BaseModel

class GenerateRequest(BaseModel):
    model: str   # "gpt", "flan", or "gpu"
    prompt: str
    max_new_tokens: int | None = None
