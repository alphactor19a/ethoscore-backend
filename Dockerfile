# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system dependencies including git and git-lfs
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a user to run the application (security best practice)
# But for now, we run as root to ensure permission to write/download models
# If you want to run as a user, ensure the user has write access to /app

# Expose the port
EXPOSE $PORT

# Run the application
# We rely on the app's internal startup logic to download models if they are missing
# This keeps the build size smaller and faster
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT"]

