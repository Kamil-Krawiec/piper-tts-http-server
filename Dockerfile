# Base image: Python 3.11 Slim (Debian based).
# Pinned to an exact digest for reproducibility and supply-chain integrity.
# To bump the base image, replace the digest below with the output of
# `docker pull python:3.11-slim` (keep the digest pinned).
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

# Install system dependencies.
# `apt-get upgrade` is deliberate: the official base image freezes Debian package
# versions on the day it is published, so without it the image ships ~800+ known
# OS CVEs (Trivy). Upgrading at build time pulls the current security patches.
# `--no-install-recommends` avoids pulling optional packages (smaller attack surface).
# espeak-ng is CRITICAL for Piper; ffmpeg enables mp3 conversion.
# ca-certificates is required for HTTPS model downloads via requests/curl.
# DEBIAN_FRONTEND is set so dpkg never prompts (non-interactive build).
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        espeak-ng \
        libsndfile1 \
        ffmpeg \
        ca-certificates \
        curl \
    && apt-get autoremove -y \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade the Python tooling bundled in the base image. The official python image
# ships older pip/setuptools/wheel, which carry known CVEs (e.g. CVE-2026-24049,
# CVE-2025-8869, CVE-2026-59890). Upgrading at build time keeps them out of the
# final image. This is a runtime-only image, so this does not affect the build of
# the app packages themselves.
RUN python -m pip install --no-cache-dir --upgrade \
    pip setuptools wheel

# Copy python requirements and install
# requirements.txt pins the application dependencies; the tooling upgrade above
# is what refreshes pip/setuptools/wheel without disturbing the app's pins.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    -r requirements.txt

# Copy the server code and entrypoint
COPY server.py docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Set environment variables
ENV DATA_DIR=/data
ENV PORT=5000

# Create data directory
RUN mkdir -p /data

# Expose the port
EXPOSE 5000

# Entrypoint delegates to server.py; pass --sd-activate for socket activation
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "server.py"]
