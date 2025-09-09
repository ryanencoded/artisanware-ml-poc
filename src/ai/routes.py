from fastapi import FastAPI
from .models import GenerateRequest
from .services import get_service

app = FastAPI(title="AI Microservice", version="0.5.0")

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/generate")
def generate_text(req: GenerateRequest):
    try:
        service = get_service(req.model)
    except ValueError as e:
        return {"error": str(e)}

    text = service(req.prompt, req.max_new_tokens)
    return {"generated_text": text}
