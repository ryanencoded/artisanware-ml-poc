PYTHON=python3.11
VENV=.venv

setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

export:
	. $(VENV)/bin/activate && $(PYTHON) src/ai-python/export_to_onnx.py
