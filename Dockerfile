FROM python:3.10.7

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
COPY api/ api
COPY data_models.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]