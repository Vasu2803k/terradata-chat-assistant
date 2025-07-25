FROM python:3.11-slim

WORKDIR /app

# Use pip cache for faster builds
ENV PIP_NO_CACHE_DIR=0
ENV PIP_CACHE_DIR=/tmp/pip-cache

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"] 