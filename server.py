import os
import logging
import subprocess
import requests
import uvicorn
from pathlib import Path
from typing import Optional, Generator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# --- Configuration ---
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
HF_URL_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
PORT = int(os.getenv("PORT", 5000))

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("piper-openai-tts")

# --- FastAPI App ---
app = FastAPI(
    title="Piper OpenAI TTS",
    description="A lightweight, OpenAI-compatible TTS API wrapper for Piper.",
    version="1.0.0"
)

# --- Data Models ---
class OpenAISpeechRequest(BaseModel):
    model: str = Field(..., description="Voice name (e.g., 'pl_PL-gosia-medium')")
    input: str = Field(..., description="The text to synthesize")
    voice: Optional[str] = Field(None, description="Speaker ID (for multi-speaker models)")
    response_format: Optional[str] = Field("mp3", description="Audio format.")
    speed: Optional[float] = Field(1.0, description="Speech speed.")
    
    # Piper specific parameters
    noise_scale: Optional[float] = Field(0.667, description="Noise scale")
    noise_scale_w: Optional[float] = Field(0.8, description="Phoneme noise scale")
    length_scale: Optional[float] = Field(None, description="Direct length scale")

# --- Helper Functions ---

def get_voice_paths(voice_name: str):
    return DATA_DIR / f"{voice_name}.onnx", DATA_DIR / f"{voice_name}.onnx.json"

def download_voice_if_missing(voice_name: str) -> bool:
    onnx_path, json_path = get_voice_paths(voice_name)
    if onnx_path.exists() and json_path.exists():
        return True

    logger.info(f"Voice '{voice_name}' not found locally. Downloading...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        parts = voice_name.split('-')
        if len(parts) < 3: raise ValueError("Invalid format")
        lang_code, name, quality = parts[0], parts[1], parts[2]
        lang_short = lang_code.split('_')[0]
    except Exception:
        logger.error(f"Invalid voice name format: '{voice_name}'")
        return False

    repo_path = f"{lang_short}/{lang_code}/{name}/{quality}/{voice_name}"

    try:
        for ext in [".onnx", ".onnx.json"]:
            url = f"{HF_URL_BASE}/{repo_path}{ext}"
            logger.info(f"Downloading: {url}")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(DATA_DIR / f"{voice_name}{ext}", 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        if onnx_path.exists(): onnx_path.unlink()
        if json_path.exists(): json_path.unlink()
        return False

def stream_audio_generator(cmd: list, input_text: str, output_file: str) -> Generator:
    try:
        subprocess.run(
            cmd,
            input=input_text,
            text=True,
            check=True,
            stderr=subprocess.PIPE
        )
        with open(output_file, "rb") as audio:
            yield from audio
    except subprocess.CalledProcessError as e:
        logger.error(f"Piper process failed: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Piper failed: {e.stderr}")
    finally:
        if os.path.exists(output_file):
            os.remove(output_file)

# --- API Endpoints ---

@app.post("/v1/audio/speech")
async def generate_speech(request: OpenAISpeechRequest):
    voice_name = request.model
    if not download_voice_if_missing(voice_name):
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found.")

    onnx_path, _ = get_voice_paths(voice_name)
    output_file = f"/tmp/piper_{os.getpid()}_{os.urandom(4).hex()}.wav"
    
    if request.length_scale:
        final_length_scale = request.length_scale
    else:
        final_length_scale = 1.0 / max(0.1, request.speed)

    # --- FIX IS HERE ---
    # Changed arguments to match Piper CLI (hyphens instead of underscores)
    cmd = [
        "piper",
        "--model", str(onnx_path),
        "--output_file", output_file,
        "--length-scale", str(final_length_scale), # Fixed
        "--noise-scale", str(request.noise_scale),   # Fixed
        "--noise-w", str(request.noise_scale_w)      # Fixed (was --noise_scale_w)
    ]

    if request.voice:
        cmd.extend(["--speaker", str(request.voice)])

    return StreamingResponse(
        stream_audio_generator(cmd, request.input, output_file),
        media_type="audio/wav"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)