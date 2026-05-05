# Base image Python
FROM python:3.11-slim

# Supaya log langsung keluar (penting di Railway)
ENV PYTHONUNBUFFERED=1

# Install system dependencies (penting untuk numpy / tensorflow)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements dulu (biar cache efisien)
COPY requirements.txt .

# Upgrade pip & install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project
COPY . .

# Railway pakai PORT env
ENV PORT=8000

# Run app dengan gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
