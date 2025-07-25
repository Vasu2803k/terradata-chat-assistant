# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies and Python dependencies in one layer for better caching
COPY requirements.txt ./
RUN apt-get update \
    && apt-get install -y build-essential \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the code (after dependencies for better cache usage)
COPY . .

# Expose backend and frontend ports
EXPOSE 8000 8501

# Default command (overridden by docker-compose)
CMD ["bash"] 