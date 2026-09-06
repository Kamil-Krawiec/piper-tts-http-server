import argparse
import asyncio
import json
import logging
import os
import requests
import socket
import subprocess
import tempfile
import threading
import uvicorn
from pathlib import Path
from typing import Optional, Generator, Union

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

# --- Configuration ---
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
HF_URL_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
PORT = int(os.getenv("PORT", 5000))

# --- Hardening limits (server-side guardrails) ---
# Bound a single synthesis request so a malicious or accidental payload cannot
# drive unbounded CPU/memory/disk use (DoS). Strings are capped at character
# level, lists at segment level, and every Piper numeric knob is range-checked.
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "50000"))
MAX_INPUT_SEGMENTS = int(os.getenv("MAX_INPUT_SEGMENTS", "200"))
MAX_NAME_LENGTH = int(os.getenv("MAX_NAME_LENGTH", "128"))

# Voice model download guardrails. A voice can be large (100+ MB), so cap the
# accepted size and never expose a partially-downloaded file as a valid voice.
MAX_VOICE_BYTES = int(os.getenv("MAX_VOICE_BYTES", "524288000"))

# Subprocess timeouts: Piper synthesis and ffmpeg conversion must never hang a
# worker thread indefinitely (resource/threadpool exhaustion).
PIPER_TIMEOUT = int(os.getenv("PIPER_TIMEOUT", "120"))
FFMPEG_TIMEOUT = int(os.getenv("FFMPEG_TIMEOUT", "60"))

# Reject request bodies above this size before they are read into memory.
# Default 2 MiB is far above any legitimate synthesis request (max input is
# 50k characters of JSON).
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", "2097152"))

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
    description="An OpenAI-compatible audio API wrapper for Piper.",
    version="1.1.0",
)


@app.middleware("http")
async def limit_request_body(request, call_next):
    """Reject oversized request bodies before they are buffered into memory."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                return JSONResponse(
                    status_code=413, content={"detail": "Request body too large."}
                )
        except ValueError:
            # Malformed Content-Length header; let the server handle it downstream.
            pass
    return await call_next(request)


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
            "response_format": "mp3",
            "speed": 1.0
        }

    Notes for this wrapper:
    - `voice` is used as Piper voice name (e.g. 'pl_PL-gosia-medium').
    - For compatibility with some clients (e.g. Open WebUI), if `voice`
      is missing, `model` will be used as the Piper voice name.
    - OpenAI uses `response_format`; this wrapper also accepts `format`
      as a convenience alias.
    """

    model: str = Field(
        ...,
        max_length=MAX_NAME_LENGTH,
        description="TTS model name (kept for OpenAI compatibility; ignored by Piper).",
    )
    input: Union[str, list[str]] = Field(
        ..., description="The text to synthesize, or list of text segments."
    )
    voice: Optional[str] = Field(
        None,
        max_length=MAX_NAME_LENGTH,
        description=(
            "Logical voice name (e.g. 'pl_PL-gosia-medium'). "
            "If omitted, `model` will be used as voice name."
        ),
    )
    # OpenAI uses `response_format`; we also accept `format` for lenient clients.
    format: Optional[str] = Field(
        None,
        description=(
            "Alias for OpenAI's `response_format`. Supported: 'wav', 'mp3'. "
            "Default: 'wav' if not set."
        ),
    )
    response_format: Optional[str] = Field(
        None,
        description="OpenAI field name for audio format (alias for `format`).",
    )
    speed: Optional[float] = Field(
        1.0,
        ge=0.25,
        le=4.0,
        description="Speech speed multiplier. 1.0 is normal speed.",
    )

    # Piper-specific optional parameters (non-standard but harmless extras)
    # Each is range-checked so extreme values cannot crash or hang Piper.
    noise_scale: Optional[float] = Field(
        0.667, ge=0.0, le=1.5, description="Piper noise scale (controls prosody variation)."
    )
    noise_scale_w: Optional[float] = Field(
        0.8, ge=0.0, le=1.5, description="Piper phoneme noise scale."
    )
    length_scale: Optional[float] = Field(
        None,
        ge=0.1,
        le=10.0,
        description=(
            "Direct Piper length scale. If omitted, derived from `speed` "
            "as length_scale = 1.0 / max(0.1, speed)."
        ),
    )
    speaker: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Optional numeric speaker ID for multi-speaker Piper models. "
            "Not part of the OpenAI spec but accepted as an extension."
        ),
    )

    @field_validator("input")
    @classmethod
    def _validate_input(cls, value: Union[str, list[str]]) -> Union[str, list[str]]:
        """Bound the synthesis payload to prevent unbounded resource use."""
        if isinstance(value, str):
            chars = len(value)
            if chars > MAX_INPUT_CHARS:
                raise ValueError(
                    f"`input` too long: {chars} characters "
                    f"(maximum {MAX_INPUT_CHARS})."
                )
        elif isinstance(value, list):
            if len(value) > MAX_INPUT_SEGMENTS:
                raise ValueError(
                    f"`input` has too many segments: {len(value)} "
                    f"(maximum {MAX_INPUT_SEGMENTS})."
                )
            total_chars = sum(len(segment) for segment in value)
            if total_chars > MAX_INPUT_CHARS:
                raise ValueError(
                    f"`input` too long: {total_chars} characters total "
                    f"(maximum {MAX_INPUT_CHARS})."
                )
        return value


# --- Helper Functions ---


def get_voice_paths(voice_name: str):
    """Return local .onnx and .onnx.json paths for a given Piper voice."""
    if (
        "/" in voice_name
        or "\\" in voice_name
        or ".." in voice_name
        or voice_name.startswith(".")
        or "\x00" in voice_name
    ):
        raise ValueError(f"Invalid voice name: {voice_name!r}")
    return DATA_DIR / f"{voice_name}.onnx", DATA_DIR / f"{voice_name}.onnx.json"


def get_installed_voice_catalog() -> list[dict[str, str]]:
    """Return locally installed, runnable Piper voices in stable order."""
    voices = []
    for onnx_path in DATA_DIR.glob("*.onnx"):
        voice_name = onnx_path.stem
        if onnx_path.is_file() and onnx_path.with_suffix(".onnx.json").is_file():
            voices.append({"id": voice_name, "name": voice_name})
    return sorted(voices, key=lambda voice: voice["id"])


# Per-voice download locks so two concurrent requests for the same (not yet
# cached) voice do not both hit Hugging Face and race writing the same files.
_voice_locks_guard = threading.Lock()
_voice_locks: dict[str, threading.Lock] = {}


def _get_voice_lock(voice_name: str) -> threading.Lock:
    with _voice_locks_guard:
        lock = _voice_locks.get(voice_name)
        if lock is None:
            lock = threading.Lock()
            _voice_locks[voice_name] = lock
        return lock


def _parse_voice_repo_path(voice_name: str) -> str:
    """Build the Hugging Face repo path for a voice, validating its structure."""
    parts = voice_name.split("-")
    if len(parts) < 3:
        raise ValueError("Invalid voice name format")
    lang_code, name, quality = parts[0], parts[1], parts[2]
    lang_short = lang_code.split("_")[0]
    return f"{lang_short}/{lang_code}/{name}/{quality}/{voice_name}"


def download_voice_if_missing(voice_name: str) -> bool:
    """
    Ensure that Piper voice files are present locally.

    Downloads are written to a temporary path and atomically moved into place
    only after they validate (non-empty model and parseable JSON metadata), so
    a partially-downloaded or corrupted voice is never exposed to Piper.
    A per-voice lock prevents concurrent duplicate downloads.
    """
    try:
        onnx_path, json_path = get_voice_paths(voice_name)
        repo_path = _parse_voice_repo_path(voice_name)
    except ValueError as exc:
        logger.error("Invalid voice name format: '%s'", voice_name)
        logger.debug("Voice validation error: %s", exc)
        return False

    with _get_voice_lock(voice_name):
        # Re-check after acquiring the lock: another thread may have downloaded.
        if onnx_path.exists() and json_path.exists():
            return True

        logger.info("Voice '%s' not found locally. Downloading...", voice_name)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        tmp_onnx = DATA_DIR / f".{voice_name}.onnx.{os.getpid()}.tmp"
        tmp_json = DATA_DIR / f".{voice_name}.onnx.json.{os.getpid()}.tmp"
        tmp_paths = (tmp_onnx, tmp_json)

        try:
            for ext, tmp_path in ((".onnx", tmp_onnx), (".onnx.json", tmp_json)):
                url = f"{HF_URL_BASE}/{repo_path}{ext}"
                logger.info("Downloading: %s", url)
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    content_length = r.headers.get("content-length")
                    if content_length and int(content_length) > MAX_VOICE_BYTES:
                        raise ValueError(
                            f"Voice exceeds size limit ({MAX_VOICE_BYTES} bytes)"
                        )
                    written = 0
                    with open(tmp_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            written += len(chunk)
                            if written > MAX_VOICE_BYTES:
                                raise ValueError(
                                    f"Voice exceeds size limit ({MAX_VOICE_BYTES} bytes)"
                                )

                # Validate the downloaded artifact before publishing it.
                if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                    raise ValueError("Downloaded voice file is empty")

            # The metadata file must be well-formed JSON.
            with open(tmp_json, "r", encoding="utf-8") as f:
                json.load(f)

            # Atomically move both files into place (same filesystem).
            os.replace(tmp_onnx, onnx_path)
            os.replace(tmp_json, json_path)
            return True
        except (requests.RequestException, ValueError, OSError) as e:
            logger.error("Failed to download voice '%s': %s", voice_name, e)
            for tmp in tmp_paths:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
            return False


def _remove_quietly(path: str) -> None:
    """Best-effort removal of a file, ignoring errors (used for temp cleanup)."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError as e:
        logger.warning("Failed to remove temporary file '%s': %s", path, e)


def normalize_audio_format(target_format: Optional[str]) -> tuple[str, str]:
    """Validate/normalize audio format and return (format, media_type)."""
    fmt = (target_format or "wav").lower()
    if fmt == "wav":
        return "wav", "audio/wav"
    if fmt == "mp3":
        return "mp3", "audio/mpeg"
    raise HTTPException(
        status_code=400,
        detail="Unsupported audio format. Only 'wav' and 'mp3' are supported.",
    )


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
    target_format, media_type = normalize_audio_format(target_format)

    if target_format == "wav":
        return wav_path, media_type, []

    mp3_path = str(Path(wav_path).with_suffix(".mp3"))
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
            timeout=FFMPEG_TIMEOUT,
        )
        if not os.path.exists(mp3_path):
            logger.error("ffmpeg reported success but MP3 file was not created: %s", mp3_path)
            raise HTTPException(
                status_code=500,
                detail="ffmpeg did not produce the expected MP3 file. Check output path and permissions.",
            )
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg conversion timed out after %ss", FFMPEG_TIMEOUT)
        _remove_quietly(mp3_path)
        raise HTTPException(
            status_code=500,
            detail="ffmpeg conversion timed out. The audio may be too complex to encode.",
        )
    except subprocess.CalledProcessError as e:
        # Remove any partially-written MP3 so it cannot be treated as valid output.
        _remove_quietly(mp3_path)
        logger.error("ffmpeg conversion failed: %s", e.stderr.decode("utf-8", "ignore"))
        raise HTTPException(
            status_code=500,
            detail="Failed to convert audio to mp3. Is ffmpeg installed in the container?",
        )
    

    return mp3_path, media_type, [mp3_path]


def stream_audio_generator(
    piper_cmd: list[str],
    input_text: str,
    wav_path: str,
    target_format: str,
) -> Generator[bytes, None, None]:
    """
    Run Piper to produce a WAV file, optionally convert it, then stream audio bytes.
    """
    cleanup_files = [wav_path]

    try:
        # 1) Run Piper to produce WAV file
        try:
            result = subprocess.run(
                piper_cmd,
                input=input_text,
                text=True,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=PIPER_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.error("Piper synthesis timed out after %ss", PIPER_TIMEOUT)
            raise HTTPException(
                status_code=500,
                detail="Audio synthesis timed out. The request may be too large.",
            )

        if result.returncode != 0:
            logger.error("Piper process failed: %s", result.stderr)
            raise HTTPException(
                status_code=500,
                detail=f"Piper failed with exit code {result.returncode}",
            )

        # 2) Now that WAV exists, perform optional conversion
        final_path, _, extra_cleanup = convert_audio_if_needed(
            wav_path, target_format
        )
        if extra_cleanup:
            cleanup_files.extend(extra_cleanup)

        # 3) Stream the final audio file (WAV or MP3)
        with open(final_path, "rb") as audio:
            for chunk in iter(lambda: audio.read(8192), b""):
                yield chunk
    finally:
        # 4) Cleanup temporary files
        for f in cleanup_files:
            _remove_quietly(f)


# --- API Endpoints ---


@app.post("/audio/speech", include_in_schema=False)
@app.post("/v1/audio/speech")
async def generate_speech(request: OpenAISpeechRequest):
    """
    OpenAI-compatible TTS endpoint.

    The handler stays async for streaming compatibility, but the blocking voice
    download (network I/O + large file writes) is dispatched to a worker thread
    via run_in_threadpool so it never stalls the asyncio event loop. Piper and
    ffmpeg run inside the streaming generator, which Starlette also runs in a
    worker thread.

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

    try:
        onnx_path, _ = get_voice_paths(voice_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Resolve effective output format (response_format has precedence over format)
    requested_format = request.response_format or request.format or "wav"
    target_format, media_type = normalize_audio_format(requested_format)

    # Normalize and join input text
    if isinstance(request.input, list):
        input_text = "\n".join(request.input)
    else:
        input_text = request.input

    if not input_text:
        raise HTTPException(status_code=400, detail="`input` text must not be empty.")

    logger.info(
        "Speech request parameters: model=%r requested_voice=%r "
        "effective_voice=%r format=%r response_format=%r effective_format=%r "
        "speed=%s noise_scale=%s noise_scale_w=%s length_scale=%s speaker=%s "
        "input_segments=%d input_characters=%d",
        request.model,
        request.voice,
        voice_name,
        request.format,
        request.response_format,
        target_format,
        request.speed,
        request.noise_scale,
        request.noise_scale_w,
        request.length_scale,
        request.speaker,
        len(request.input) if isinstance(request.input, list) else 1,
        len(input_text),
    )

    # Ensure voice files exist. This does network I/O + large file writes, so it
    # is offloaded to a worker thread rather than run on the event loop.
    if not await run_in_threadpool(download_voice_if_missing, voice_name):
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found.")

    # Secure temporary file for Piper output (O_EXCL, random name). The file is
    # created empty and Piper truncates/overwrites it via wave.open(..., "wb").
    wav_fd, wav_path = tempfile.mkstemp(prefix="piper_", suffix=".wav")
    os.close(wav_fd)

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

    return StreamingResponse(
        stream_audio_generator(
            piper_cmd=piper_cmd,
            input_text=input_text,
            wav_path=wav_path,
            target_format=target_format,
        ),
        media_type=media_type,
    )


@app.get("/audio/voices", include_in_schema=False)
@app.get("/v1/audio/voices")
async def list_voices():
    """List installed Piper voices for OpenAI-compatible clients."""
    return {"voices": get_installed_voice_catalog()}


@app.get("/audio/models", include_in_schema=False)
@app.get("/v1/audio/models")
async def list_models():
    """List installed Piper voice models for OpenAI-compatible clients."""
    return {"models": get_installed_voice_catalog()}


class _ActivatedSocket(socket.socket):
    """Socket that skips listen() — already done by systemd."""

    def listen(self, __backlog: int = 0) -> None:  # noqa: D102
        pass


def _run_sd_activated() -> None:
    """Start uvicorn using a systemd socket-activated file descriptor.

    systemd passes the listening socket as fd 3.  uvicorn 0.27 hardcodes
    AF_UNIX in socket.fromfd() for --fd, and asyncio calls sock.listen()
    which fails on an already-listening socket in rootless containers.
    We work around both issues with a custom socket subclass.
    """
    sock = _ActivatedSocket(
        fileno=3,
        family=socket.AF_INET6,
        type=socket.SOCK_STREAM,
    )
    sock.setblocking(False)

    config = uvicorn.Config("server:app")
    server = uvicorn.Server(config)
    asyncio.run(server.serve(sockets=[sock]))


def _run_http_server() -> None:
    """Start uvicorn binding directly to host:port."""
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piper OpenAI TTS HTTP Server")
    parser.add_argument(
        "--sd-activate",
        action="store_true",
        help="Use systemd socket activation (fd 3) instead of binding a port.",
    )
    args = parser.parse_args()

    if args.sd_activate:
        _run_sd_activated()
    else:
        _run_http_server()
