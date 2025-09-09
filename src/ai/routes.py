from fastapi import FastAPI
from .models import GenerateRequest
from .services.gpt import generate_with_gpt
from .services.flan import generate_with_flan
from .services.gpu import generate_with_gpu

app = FastAPI(title="AI Microservice", version="0.4.0")

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/generate")
def generate_text(req: GenerateRequest):
    if req.model == "gpt":
        text = generate_with_gpt(req.prompt, req.max_new_tokens)
    elif req.model == "flan":
        text = generate_with_flan(req.prompt, req.max_new_tokens)
    elif req.model == "gpu":
        text = generate_with_gpu(req.prompt, "default", req.max_new_tokens)
    else:
        return {"error": "Invalid model. Use 'gpt', 'flan', or 'gpu'."}

    return {"generated_text": text}
