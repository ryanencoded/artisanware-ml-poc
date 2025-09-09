PYTHON=python3.11
VENV=.venv

install-ai:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

start-ai:
	uvicorn src.ai.routes:app --reload