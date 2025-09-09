import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load env vars
MODEL_ID = os.getenv("GPU_MODEL_ID", "microsoft/phi-2")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))

# Pick device: use Apple GPU if available, else CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load tokenizer + model once
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)

def generate_with_gpu(prompt: str, max_new_tokens: int | None = None) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.95,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
