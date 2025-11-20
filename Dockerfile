# Base image: Python 3.11 Slim (Debian based)
FROM python:3.11-slim

# Install system dependencies
# espeak-ng is CRITICAL for Piper
RUN apt-get update && apt-get install -y \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server code
COPY server.py .

# Set environment variables
ENV DATA_DIR=/data
ENV PORT=5000

# Create data directory
RUN mkdir -p /data

# Expose the port
EXPOSE 5000

# Run the server using python directly
CMD ["python", "server.py"]