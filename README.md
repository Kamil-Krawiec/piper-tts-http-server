# Piper OpenAI TTS HTTP Server

A lightweight FastAPI service that wraps [Piper](https://github.com/OHF-Voice/piper1-gpl) in an OpenAI-compatible `/v1/audio/speech` endpoint.  
It can be run locally or as a Docker container and automatically downloads Rhasspy Piper voices on demand.

## Highlights
- **Drop-in HTTP API** &ndash; mimics the OpenAI Audio Speech endpoint so you can reuse existing OpenAI client code.
- **Self-hosted voices** &ndash; ships with the official `piper-tts` Python package and pulls voices from Hugging Face (`rhasspy/piper-voices`).
- **Docker ready** &ndash; published to Docker Hub as `kod-zero/piper-openai-tts`, with a GitHub Actions workflow for automatic pushes.
- **Configurable synthesis** &ndash; control model, speaker, speech speed, noise, and length-scale per request.

---

## Getting Started

### Local Python environment
```bash
git clone https://github.com/your-user/piper-tts-http-server.git
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
  kod-zero/piper-openai-tts:latest
```

Need to customize the image? Build it locally:
```bash
docker build -t my-piper-tts .
docker run --rm -p 5000:5000 -v "$(pwd)/piper-data:/data" my-piper-tts
```

---

## API Overview

- **Endpoint:** `POST /v1/audio/speech`
- **Response:** Streams back synthesized audio (`audio/wav`)
- **Voice assets:** Downloaded from `https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0`

### Request body

| Field            | Type    | Required | Default | Description |
|------------------|---------|----------|---------|-------------|
| `model`          | string  | ✅       | —       | Piper voice name, e.g. `pl_PL-gosia-medium`. Determines which ONNX voice file to use. |
| `input`          | string  | ✅       | —       | Text to synthesize. |
| `voice`          | string  | ❌       | `null`  | Speaker ID for multi-speaker voices. |
| `response_format`| string  | ❌       | `mp3`   | Included for OpenAI compatibility. Output is currently WAV. |
| `speed`          | float   | ❌       | `1.0`   | Speaking rate (higher = faster). Controls Piper length scale. |
| `noise_scale`    | float   | ❌       | `0.667` | Controls pronunciation variability. |
| `noise_scale_w`  | float   | ❌       | `0.8`   | Phoneme-level noise. |
| `length_scale`   | float   | ❌       | derived | Optional direct Piper `--length-scale`. Overrides `speed`. |

### Example (Python)

```python
import requests

payload = {
    "model": "pl_PL-gosia-medium",
    "input": "Cześć! To jest demo syntezy mowy Piper.",
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
  -d '{"model":"en_US-lessac-high","input":"Hello from Piper!"}' \
  http://localhost:5000/v1/audio/speech
```

---

## Voice Management
- Voice names follow the Piper naming scheme `<lang>_<REGION>-<voice>-<quality>`, for example `en_US-amy-low`.  
- When a voice is requested for the first time, the server downloads `<voice>.onnx` and `<voice>.onnx.json` to `DATA_DIR`.
- To preload voices, place both files in `DATA_DIR` ahead of time.
- Remove cached voices by deleting their ONNX/JSON files; they will be re-downloaded on the next request.

---

## Configuration & Customization

| Environment variable | Default | Purpose |
|----------------------|---------|---------|
| `DATA_DIR`           | `/data` | Location where voice files are cached. Mount/persist this when using Docker. |
| `PORT`               | `5000`  | Port that Uvicorn binds to. |

Additional tweaks:
- **Custom Hugging Face mirror**: edit `HF_URL_BASE` in `server.py` if you host the voice files elsewhere.
- **Alternative Piper args**: modify `cmd` in `server.py` to include extra flags (e.g., `--speaker-id` for experimental features).
- **Logging**: `logging.basicConfig` is defined at the top of `server.py`; adjust the level or format to suit your deployment.

---

## How It Works
1. FastAPI receives the request and validates it with Pydantic.
2. The server ensures the requested voice exists locally, downloading it from Hugging Face if missing.
3. Piper is invoked via the `piper` CLI (shipped through the `piper-tts` dependency) with the requested parameters.
4. The resulting WAV file is streamed back to the client and then deleted from `/tmp`.

Dependencies are declared in `requirements.txt` (FastAPI, Uvicorn, Requests, `piper-tts`, Pydantic) and system packages (`espeak-ng`, `libsndfile1`) are provided by the Docker image.

---

## Contributing & Support
- Issues and pull requests are welcome for bug fixes, new documentation, or feature enhancements.
- The project is licensed under the [MIT License](LICENSE).
- For reproducible Docker builds, see `.github/workflows/publish_docker_image_to_hub.yml`.

Happy building & enjoy self-hosted Piper speech synthesis!
