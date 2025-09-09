PYTHON=python3.11
VENV=.venv
MODEL_NAME=flan-t5-base         # short name only
MODEL_HF_ID=google/flan-t5-base # full Hugging Face repo id
MODEL_TASK=text2text-generation
MODEL_DIR="models/$(MODEL_NAME)"
ONNX_RAW_DIR="$(MODEL_DIR)/onnx-raw"
ONNX_DIR="$(MODEL_DIR)/onnx"

setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

clean:
	rm -rf $(MODEL_DIR)

start-ai-app:
	uvicorn src.ai.routes:app --reload

model-files:
	mkdir -p $(MODEL_DIR)
	curl -L https://huggingface.co/$(MODEL_HF_ID)/resolve/main/config.json -o $(MODEL_DIR)/config.json
	curl -L https://huggingface.co/$(MODEL_HF_ID)/resolve/main/generation_config.json -o $(MODEL_DIR)/generation_config.json || true
	curl -L https://huggingface.co/$(MODEL_HF_ID)/resolve/main/tokenizer.json -o $(MODEL_DIR)/tokenizer.json
	curl -L https://huggingface.co/$(MODEL_HF_ID)/resolve/main/tokenizer_config.json -o $(MODEL_DIR)/tokenizer_config.json
	curl -L https://huggingface.co/$(MODEL_HF_ID)/resolve/main/special_tokens_map.json -o $(MODEL_DIR)/special_tokens_map.json || true
	curl -L https://huggingface.co/$(MODEL_HF_ID)/resolve/main/spiece.model -o $(MODEL_DIR)/spiece.model || true
	curl -L https://huggingface.co/$(MODEL_HF_ID)/resolve/main/vocab.json -o $(MODEL_DIR)/vocab.json || true
	curl -L https://huggingface.co/$(MODEL_HF_ID)/resolve/main/merges.txt -o $(MODEL_DIR)/merges.txt || true

model-export:
	. $(VENV)/bin/activate && optimum-cli export onnx \
		--model $(MODEL_HF_ID) \
		--task $(MODEL_TASK) \
		--framework pt \
		$(ONNX_RAW_DIR)

model-quantize:
	. $(VENV)/bin/activate && optimum-cli onnxruntime quantize \
		--onnx_model $(ONNX_RAW_DIR) \
		--output $(ONNX_DIR) \
		--avx2
	# Seq2seq models expect a merged decoder
	cp $(ONNX_DIR)/decoder_model_quantized.onnx $(ONNX_DIR)/decoder_model_merged_quantized.onnx || true

env-file:
	@echo "MODEL_NAME=$(MODEL_NAME)" > .env
	@echo "MODEL_TASK=$(MODEL_TASK)" >> .env
	@echo "MODEL_DIR=$(MODEL_DIR)" >> .env

model-prepare: clean model-export model-quantize model-files
