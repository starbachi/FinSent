# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY config/ config/
COPY models/ models/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash finsent
RUN chown -R finsent:finsent /app
USER finsent

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# TODO: Configure entry point
# Options for different services:

# API Service
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Training Service
# CMD ["python", "src/models/train.py"]

# Data Processing Service  
# CMD ["python", "src/data/process.py"]

# TODO: Multi-stage build optimization
# Consider using multi-stage builds to reduce image size:
# 
# FROM python:3.9-slim as builder
# # Install build dependencies and create wheels
# 
# FROM python:3.9-slim as runtime
# # Copy only necessary files from builder
