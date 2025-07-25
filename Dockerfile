# Use official Python image
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies with pip cache
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the rest of the code
COPY . .

EXPOSE 8000 8501
CMD ["bash"]
    