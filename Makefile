setup:
	python -m pip install -r requirements.txt

data:
	python generate_mock_data.py

train:
	python train.py

api:
	uvicorn api.main:app --reload

ui:
	streamlit run app/streamlit_app.py

test:
	pytest -q

all:
	python generate_mock_data.py
	python train.py
	pytest -q
