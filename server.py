import os
import logging
import subprocess
import requests
import uvicorn
from pathlib import Path
from typing import Optional, Generator, Union, List

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
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("piper-openai-tts")

# --- FastAPI App ---
app = FastAPI(
    title="Piper OpenAI TTS",
    description="An OpenAI-compatible /v1/audio/speech API wrapper for Piper.",
    version="1.1.0",
)


# --- Data Models ---
class OpenAISpeechRequest(BaseModel):
    """
    Request body compatible with OpenAI's /v1/audio/speech endpoint.

    Reference shape (simplified):

        POST /v1/audio/speech
        {
            "model": "gpt-4o-mini-tts",
            "input": "Hello world",
            "voice": "alloy",
            "format": "mp3",
            "speed": 1.0
        }

    Notes for this wrapper:
    - `voice` is used as Piper voice name (e.g. 'pl_PL-gosia-medium').
    - For compatibility with some clients (e.g. Open WebUI), if `voice`
      is missing, `model` will be used as the Piper voice name.
    """

    model: str = Field(
        ...,
        description="TTS model name (kept for OpenAI compatibility; ignored by Piper).",
    )
    input: Union[str, List[str]] = Field(
        ..., description="The text to synthesize, or list of text segments."
    )
    voice: Optional[str] = Field(
        None,
        description=(
            "Logical voice name (e.g. 'pl_PL-gosia-medium'). "
            "If omitted, `model` will be used as voice name."
        ),
    )
    # OpenAI uses `format`; we also accept legacy `response_format` for lenient clients.
    format: Optional[str] = Field(
        None,
        description="Output audio format. Supported: 'wav', 'mp3'. Default: 'wav' if not set.",
    )
    response_format: Optional[str] = Field(
        None,
        description="Deprecated alias for `format` (accepted for compatibility).",
    )
    speed: Optional[float] = Field(
        1.0,
        ge=0.25,
        le=4.0,
        description="Speech speed multiplier. 1.0 is normal speed.",
    )

    # Piper-specific optional parameters (non-standard but harmless extras)
    noise_scale: Optional[float] = Field(
        0.667, description="Piper noise scale (controls prosody variation)."
    )
    noise_scale_w: Optional[float] = Field(
        0.8, description="Piper phoneme noise scale."
    )
    length_scale: Optional[float] = Field(
        None,
        description=(
            "Direct Piper length scale. If omitted, derived from `speed` "
            "as length_scale = 1.0 / max(0.1, speed)."
        ),
    )
    speaker: Optional[int] = Field(
        None,
        description=(
            "Optional numeric speaker ID for multi-speaker Piper models. "
            "Not part of the OpenAI spec but accepted as an extension."
        ),
    )


# --- Helper Functions ---


def get_voice_paths(voice_name: str):
    """Return local .onnx and .onnx.json paths for a given Piper voice."""
    return DATA_DIR / f"{voice_name}.onnx", DATA_DIR / f"{voice_name}.onnx.json"


def download_voice_if_missing(voice_name: str) -> bool:
    """
    Ensure that Piper voice files are present locally.

    Voice naming convention is assumed to follow Rhasspy / Piper repos, e.g.:
    pl_PL-gosia-medium -> pl/pl_PL/gosia/medium/pl_PL-gosia-medium.{onnx,onnx.json}
    """
    onnx_path, json_path = get_voice_paths(voice_name)

    if onnx_path.exists() and json_path.exists():
        return True

    logger.info("Voice '%s' not found locally. Downloading...", voice_name)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        parts = voice_name.split("-")
        if len(parts) < 3:
            raise ValueError("Invalid voice name format")
        lang_code, name, quality = parts[0], parts[1], parts[2]
        lang_short = lang_code.split("_")[0]
    except Exception:
        logger.error("Invalid voice name format: '%s'", voice_name)
        return False

    repo_path = f"{lang_short}/{lang_code}/{name}/{quality}/{voice_name}"

    try:
        for ext in (".onnx", ".onnx.json"):
            url = f"{HF_URL_BASE}/{repo_path}{ext}"
            logger.info("Downloading: %s", url)
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with open(DATA_DIR / f"{voice_name}{ext}", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        logger.error("Failed to download voice '%s': %s", voice_name, e)
        if onnx_path.exists():
            onnx_path.unlink()
        if json_path.exists():
            json_path.unlink()
        return False


def convert_audio_if_needed(
    wav_path: str, target_format: str
) -> tuple[str, str, list[str]]:
    """
    Convert WAV to the requested format if necessary.

    Returns:
        (final_path, media_type, extra_cleanup_files)

    Notes:
        - Requires `ffmpeg` in PATH for mp3 conversion.
        - We only support 'wav' and 'mp3' to stay simple and predictable.
    """
    target_format = (target_format or "wav").lower()

    if target_format == "wav":
        return wav_path, "audio/wav", []

    if target_format == "mp3":
        mp3_path = wav_path.replace(".wav", ".mp3")
        cmd = [
            "ffmpeg",
            "-y",  # overwrite without prompt
            "-i",
            wav_path,
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "4",
            mp3_path,
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            logger.error("ffmpeg conversion failed: %s", e.stderr.decode("utf-8", "ignore"))
            raise HTTPException(
                status_code=500,
                detail="Failed to convert audio to mp3. Is ffmpeg installed in the container?",
            )

        return mp3_path, "audio/mpeg", [mp3_path]

    # Unsupported format -> 400 Bad Request
    raise HTTPException(
        status_code=400,
        detail="Unsupported audio format. Only 'wav' and 'mp3' are supported.",
    )


def stream_audio_generator(
    piper_cmd: list,
    input_text: str,
    wav_path: str,
    final_path: str,
    extra_cleanup: Optional[list[str]] = None,
) -> Generator[bytes, None, None]:
    """
    Run Piper, optionally convert the output, then stream audio bytes.

    This function is executed in a threadpool by Starlette, so blocking calls
    (subprocess, file I/O) are acceptable here.
    """
    cleanup_files = [wav_path]
    if extra_cleanup:
        cleanup_files.extend(extra_cleanup)

    try:
        # 1) Run Piper to produce WAV file
        result = subprocess.run(
            piper_cmd,
            input=input_text,
            text=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            logger.error("Piper process failed: %s", result.stderr)
            raise HTTPException(
                status_code=500,
                detail=f"Piper failed with exit code {result.returncode}",
            )

        # 2) If final_path != wav_path conversion has already been done
        #    outside this generator; we just stream the final file.
        path_to_stream = final_path

        with open(path_to_stream, "rb") as audio:
            for chunk in iter(lambda: audio.read(8192), b""):
                yield chunk
    finally:
        for f in cleanup_files:
            try:
                if f and os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                logger.warning("Failed to remove temporary file '%s': %s", f, e)


# --- API Endpoints ---


@app.post("/v1/audio/speech")
async def generate_speech(request: OpenAISpeechRequest):
    """
    OpenAI-compatible TTS endpoint.

    Key behaviors:
    - `voice` is required by the spec; if missing, `model` is used as a fallback
      to keep compatibility with tools that send the voice name in `model`.
    - `input` may be a string or a list of strings; lists are joined with newlines.
    - `format` / `response_format` supports 'wav' and 'mp3'.
    """
    # Determine voice name (Piper voice)
    voice_name = request.voice or request.model
    if not voice_name:
        raise HTTPException(
            status_code=400,
            detail="`voice` (or `model` as fallback) must be provided as Piper voice name.",
        )

    # Resolve effective output format (format has precedence over response_format)
    requested_format = request.format or request.response_format or "wav"

    # Normalize and join input text
    if isinstance(request.input, list):
        input_text = "\n".join(request.input)
    else:
        input_text = request.input

    if not input_text:
        raise HTTPException(status_code=400, detail="`input` text must not be empty.")

    # Ensure voice files exist
    if not download_voice_if_missing(voice_name):
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found.")

    onnx_path, _ = get_voice_paths(voice_name)

    # Temporary file paths
    wav_path = f"/tmp/piper_{os.getpid()}_{os.urandom(4).hex()}.wav"

    # Derive Piper length scale from speed if not explicitly set
    if request.length_scale is not None:
        final_length_scale = request.length_scale
    else:
        final_length_scale = 1.0 / max(0.1, (request.speed or 1.0))

    # Build Piper command
    piper_cmd = [
        "piper",
        "--model",
        str(onnx_path),
        "--output_file",
        wav_path,
        "--length-scale",
        str(final_length_scale),
        "--noise-scale",
        str(request.noise_scale),
        "--noise-w",
        str(request.noise_scale_w),
    ]

    # Optional speaker ID (multi-speaker models only)
    if request.speaker is not None:
        piper_cmd.extend(["--speaker", str(request.speaker)])

    # Perform potential format conversion up-front to know final_path + media_type
    final_path, media_type, extra_cleanup = convert_audio_if_needed(
        wav_path, requested_format
    )

    return StreamingResponse(
        stream_audio_generator(
            piper_cmd=piper_cmd,
            input_text=input_text,
            wav_path=wav_path,
            final_path=final_path,
            extra_cleanup=extra_cleanup,
        ),
        media_type=media_type,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)