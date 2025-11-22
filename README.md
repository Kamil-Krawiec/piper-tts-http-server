# Piper OpenAI TTS HTTP Server

A lightweight FastAPI service that wraps [Piper](https://github.com/OHF-Voice/piper1-gpl) in an OpenAI-compatible `/v1/audio/speech` endpoint.  
Run it locally or in Docker; it auto-downloads Rhasspy Piper voices on demand and streams audio back like the OpenAI API.

## Why self-host this
- Keep your existing OpenAI client code while owning the audio pipeline (no vendor lock-in, works offline once voices are cached).
- Cheap, local inference with Piper voices you control; ideal for on-prem, air-gapped, or cost-sensitive setups.
- Simple drop-in: POST the same payload shape as `/v1/audio/speech`; `voice` can be passed explicitly or inferred from `model`.
- Caches voices on disk so repeated requests are instant; swap or preload voices as files without code changes.
- Small, transparent stack (FastAPI + piper-tts); easy to extend Dockerfile or `server.py` if you need custom flags.

## Highlights
- **Drop-in HTTP API** &ndash; mimics the OpenAI Audio Speech endpoint so you can reuse existing OpenAI client code.
- **Self-hosted voices** &ndash; ships with the official `piper-tts` Python package and pulls voices from Hugging Face (`rhasspy/piper-voices`).
- **Docker ready** &ndash; published to Docker Hub as `kamilkrawiec/piper-openai-tts`, with a GitHub Actions workflow for automatic pushes.
- **Configurable synthesis** &ndash; control model, speaker, speech speed, noise, and length-scale per request.

---

## Getting Started

### Local Python environment
Install system dependencies (Debian/Ubuntu example):
```bash
sudo apt-get update && sudo apt-get install -y espeak-ng libsndfile1 ffmpeg  # ffmpeg only if you want MP3 output
```

```bash
git clone https://github.com/Kamil-Krawiec/piper-tts-http-server.git
cd piper-tts-http-server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
DATA_DIR=./data PORT=5000 python server.py
```

The server exposes `http://localhost:5000/v1/audio/speech` and stores/downloads Piper voice files under `DATA_DIR` (defaults to `/data`).

### Docker (recommended)

Pull the pre-built image from Docker Hub and mount a persistent data directory for cached voices:
```bash
docker run --rm \
  -p 5000:5000 \
  -e PORT=5000 \
  -v "$(pwd)/piper-data:/data" \
  kamilkrawiec/piper-openai-tts
```

Need to customize the image? Build it locally:
```bash
docker build -t my-piper-tts .
docker run --rm -p 5000:5000 -v "$(pwd)/piper-data:/data" my-piper-tts
```

The published image already contains `espeak-ng`, `libsndfile1`, and `ffmpeg` for MP3 conversion.

---

## API Overview

- **Endpoint:** `POST /v1/audio/speech`
- **Response:** Streams synthesized audio (`audio/wav` by default; `audio/mpeg` if `mp3` is requested and `ffmpeg` is installed)
- **Voice assets:** Downloaded from `https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0`

### Request body

| Field                      | Type                 | Required | Default | Description |
|----------------------------|----------------------|----------|---------|-------------|
| `model`                    | string               | ✅       | —       | OpenAI-style model name. If `voice` is omitted, this value is used as the Piper voice name (e.g., `pl_PL-gosia-medium`). |
| `voice`                    | string               | ❌       | `null`  | Preferred place to pass the Piper voice name. |
| `input`                    | string or string[]   | ✅       | —       | Text to synthesize. Lists are joined with newlines. |
| `format` / `response_format` | string             | ❌       | `wav`   | `wav` or `mp3`. MP3 conversion requires `ffmpeg` in the image/host. `format` is accepted as an alias for more lenient clients. |
| `speed`                    | float                | ❌       | `1.0`   | Speaking rate multiplier; used to derive Piper length scale. |
| `noise_scale`              | float                | ❌       | `0.667` | Piper noise scale (prosody variation). |
| `noise_scale_w`            | float                | ❌       | `0.8`   | Piper phoneme noise. |
| `length_scale`             | float                | ❌       | `1 / max(0.1, speed)` | Optional direct Piper `--length-scale`; overrides `speed` if set. |
| `speaker`                  | integer              | ❌       | `null`  | Optional speaker ID for multi-speaker voices. |

### Example (Python)

```python
import requests

payload = {
    "model": "piper",  # OpenAI compatibility field; voice selection happens via `voice`
    "voice": "pl_PL-gosia-medium",
    "input": [
        "Cześć! To jest demo syntezy mowy Piper.",
        "List inputs are joined with newlines.",
    ],
    "speed": 1.05,
    "noise_scale": 0.6
}

r = requests.post("http://localhost:5000/v1/audio/speech", json=payload, stream=True)
with open("output.wav", "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
```

### Example (curl)

```bash
curl -o output.wav \
  -H "Content-Type: application/json" \
  -d '{"model":"piper","voice":"en_US-lessac-high","input":"Hello from Piper!"}' \
  http://localhost:5000/v1/audio/speech
```

---

## Voice Management
- Voice names follow the Piper naming scheme `<lang>_<REGION>-<voice>-<quality>`, for example `en_US-amy-low`.  
- When a voice is requested for the first time, the server downloads `<voice>.onnx` and `<voice>.onnx.json` to `DATA_DIR`.
- To preload voices, place both files in `DATA_DIR` ahead of time.
- Remove cached voices by deleting their ONNX/JSON files; they will be re-downloaded on the next request.
- Voice names are validated to prevent path traversal, so stick to canonical names from `rhasspy/piper-voices`.

---

## Configuration & Customization

| Environment variable | Default | Purpose |
|----------------------|---------|---------|
| `DATA_DIR`           | `/data` | Location where voice files are cached. Mount/persist this when using Docker. |
| `PORT`               | `5000`  | Port that Uvicorn binds to. |

Additional tweaks:
- **Custom Hugging Face mirror**: edit `HF_URL_BASE` in `server.py` if you host the voice files elsewhere.
- **Alternative Piper args**: adjust how `piper_cmd` is built in `server.py` to include extra Piper flags (e.g., `--speaker`).
- **MP3 output**: `ffmpeg` is included in the Docker image; if running locally, ensure it is installed so `response_format="mp3"` succeeds.
- **Logging**: `logging.basicConfig` is defined at the top of `server.py`; adjust the level or format to suit your deployment.

---

## How It Works
1. FastAPI receives the request and validates it with Pydantic.
2. The server ensures the requested voice exists locally, downloading it from Hugging Face if missing.
3. Piper is invoked via the `piper` CLI (shipped through the `piper-tts` dependency) with the requested parameters.
4. The resulting audio is streamed back (converted to MP3 via `ffmpeg` if requested), and temporary files are cleaned up from `/tmp`.

Dependencies are declared in `requirements.txt` (FastAPI, Uvicorn, Requests, `piper-tts`, Pydantic). System packages (`espeak-ng`, `libsndfile1`, `ffmpeg`) are provided by the Docker image; if running outside Docker, install the same packages locally for best parity.

---

## Contributing & Support
- Issues and pull requests are welcome for bug fixes, new documentation, or feature enhancements.
- The project is licensed under the [MIT License](LICENSE).
- For reproducible Docker builds, see `.github/workflows/publish_docker_image_to_hub.yml`.

Happy building & enjoy self-hosted Piper speech synthesis!
