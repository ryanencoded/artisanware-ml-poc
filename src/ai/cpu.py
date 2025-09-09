import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSeq2SeqLM

# Load environment variables
load_dotenv()

# Model IDs and config
GPT_MODEL_ID = os.getenv("GPT_MODEL_ID", "HuggingFaceTB/SmolLM2-135M-Instruct")
FLAN_MODEL_ID = os.getenv("FLAN_MODEL_ID", "google/flan-t5-base")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))

# Load GPT-style model (causal LM)
gpt_tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL_ID)
gpt_model = ORTModelForCausalLM.from_pretrained(GPT_MODEL_ID, export=True)

# Load Flan/T5 model (seq2seq LM)
flan_tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL_ID)
flan_model = ORTModelForSeq2SeqLM.from_pretrained(FLAN_MODEL_ID, export=True)


def generate_with_gpt(prompt: str, max_new_tokens: int | None = None) -> str:
    inputs = gpt_tokenizer(prompt, return_tensors="pt")
    outputs = gpt_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
        num_beams=1,
        do_sample=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
        temperature=0.8,
        top_p=0.95,
        early_stopping=True,
    )
    return gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)


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
