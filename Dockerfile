# Base image: Python 3.11 Slim (Debian based).
# Pin the digest so rebuilds cannot silently change the base image.
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

# Install only runtime dependencies and apply current Debian security updates.
# espeak-ng is required by Piper; ffmpeg enables MP3 output.
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        espeak-ng \
        libsndfile1 \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep the HTTP service and its downloaded model cache out of root.
RUN useradd --system --uid 10001 --create-home --home-dir /home/piper \
        --shell /usr/sbin/nologin piper \
    && mkdir -p /data \
    && chown piper:piper /data

WORKDIR /app

# Keep Python tooling current without retaining pip's download cache.
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=piper:piper server.py docker-entrypoint.sh ./
RUN chmod 0555 docker-entrypoint.sh

ENV DATA_DIR=/data
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

USER piper
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/healthz', timeout=3)"

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "server.py"]
