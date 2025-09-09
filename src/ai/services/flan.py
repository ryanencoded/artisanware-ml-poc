import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

load_dotenv()

FLAN_MODEL_ID = os.getenv("FLAN_MODEL_ID", "google/flan-t5-base")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))

# Load once
flan_tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL_ID)
flan_model = ORTModelForSeq2SeqLM.from_pretrained(FLAN_MODEL_ID, export=True)

def generate_with_flan(prompt: str, max_new_tokens: int | None = None) -> str:
    inputs = flan_tokenizer(prompt, return_tensors="pt")
    outputs = flan_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        temperature=0.8,
        top_p=0.95,
        early_stopping=True,
    )
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
