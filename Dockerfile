# Base image: Python 3.11 Slim (Debian based), pinned to a specific digest for
# reproducibility and supply-chain integrity.
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

# Install system dependencies (non-interactive).
#   espeak-ng        required by Piper for phonemization
#   libsndfile1      audio I/O for Piper / ffmpeg
#   ffmpeg           enables MP3 output conversion
#   ca-certificates  required for HTTPS voice downloads via requests
# --no-install-recommends avoids pulling optional packages (smaller attack surface).
# apt-get upgrade pulls current security patches over the frozen base image.
# DEBIAN_FRONTEND=noninteractive keeps dpkg from prompting (build determinism).
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        espeak-ng \
        libsndfile1 \
        ffmpeg \
        ca-certificates \
    && apt-get autoremove -y \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade the Python tooling bundled in the base image. The slim image ships
# older pip/setuptools/wheel, which carry known CVEs (e.g. CVE-2026-24049 in
# wheel, CVE-2026-23949 in jaraco.context). Upgrading at build time keeps them
# out of the final image.
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server code and entrypoint
COPY server.py docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Set environment variables
ENV DATA_DIR=/data
ENV PORT=5000

# Run as an unprivileged user rather than root (container hardening).
# The data directory is created and owned by that user so voice downloads can
# write there. NOTE: if you bind-mount a host directory to /data it must be
# writable by UID 1000, e.g. `chown -R 1000:1000 ./piper-data`.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin piper \
    && mkdir -p /data \
    && chown -R piper:piper /data

USER piper

# Expose the port
EXPOSE 5000

# Healthcheck: confirm the HTTP server responds (uses the models discovery route,
# which returns 200 without authentication).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/v1/audio/models', timeout=3)" || exit 1

# Entrypoint delegates to server.py; pass --sd-activate for socket activation
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "server.py"]
