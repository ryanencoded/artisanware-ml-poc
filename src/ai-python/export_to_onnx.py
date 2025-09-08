from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import os
import argparse

MODEL_ID = "google/flan-t5-small"
OUTPUT_DIR = os.path.join("models", "flan-t5-small")

def main(force: bool = False):
    model_path = os.path.join(OUTPUT_DIR, "model.onnx")

    if os.path.exists(model_path) and not force:
        print(f"✅ Found existing ONNX model at {model_path}, skipping export.")
        return

    print(f"⬇️ Downloading and exporting {MODEL_ID} to ONNX...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = ORTModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        export=True,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"✅ Saved ONNX model + tokenizer to {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-export even if files exist")
    args = parser.parse_args()

    main(force=args.force)
