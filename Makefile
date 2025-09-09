PYTHON=python3.11
VENV=.venv

setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

start-ai-app:
	uvicorn src.ai.routes:app --reload