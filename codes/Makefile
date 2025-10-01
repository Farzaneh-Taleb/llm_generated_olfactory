install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt


format:
	black llm_response.py

lint:
	PYTHONPATH=. pylint --disable=R,C llm_response.py

all: install format lint