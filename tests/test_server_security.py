"""Security / reliability tests for the hardening changes.

These cover the guards added for request-size bounding, numeric range
validation, atomic + validated voice downloads, subprocess timeouts and
event-loop friendliness.
"""
import inspect
import json
import subprocess
from pathlib import Path
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import server


# --------------------------------------------------------------------------- #
# Request validation guards
# --------------------------------------------------------------------------- #


def test_input_too_long_rejected():
    client = TestClient(server.app)
    resp = client.post(
        "/v1/audio/speech",
        json={"model": "demo-voice", "input": "a" * (server.MAX_INPUT_CHARS + 1)},
    )
    assert resp.status_code == 422


def test_input_too_many_segments_rejected():
    client = TestClient(server.app)
    resp = client.post(
        "/v1/audio/speech",
        json={"model": "demo-voice", "input": ["x"] * (server.MAX_INPUT_SEGMENTS + 1)},
    )
    assert resp.status_code == 422


def test_model_name_too_long_rejected():
    client = TestClient(server.app)
    resp = client.post(
        "/v1/audio/speech",
        json={"model": "v" * (server.MAX_NAME_LENGTH + 1), "input": "hi"},
    )
    assert resp.status_code == 422


def test_negative_speaker_rejected():
    client = TestClient(server.app)
    resp = client.post(
        "/v1/audio/speech", json={"model": "demo-voice", "input": "hi", "speaker": -1}
    )
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "extra",
    [
        {"length_scale": 0.01},
        {"length_scale": 11.0},
        {"noise_scale": -0.5},
        {"noise_scale": 2.0},
        {"noise_scale_w": -0.1},
        {"noise_scale_w": 1.6},
        {"speed": 0.1},
        {"speed": 5.0},
    ],
)
def test_out_of_range_piper_params_rejected(extra):
    client = TestClient(server.app)
    payload = {"model": "demo-voice", "input": "hi", **extra}
    resp = client.post("/v1/audio/speech", json=payload)
    assert resp.status_code == 422, payload


def test_voice_name_with_null_byte_rejected():
    client = TestClient(server.app)
    # JSON \u0000 decodes to a real NUL byte; the voice validator must reject it.
    resp = client.post(
        "/v1/audio/speech", json={"model": "demo\u0000voice", "input": "hi"}
    )
    assert resp.status_code == 400
    assert "Invalid voice name" in resp.json()["detail"]


def test_oversized_request_body_rejected_413(monkeypatch):
    monkeypatch.setattr(server, "MAX_REQUEST_BYTES", 100)
    client = TestClient(server.app)
    body = json.dumps({"model": "demo-voice", "input": "x" * 400})
    resp = client.post(
        "/v1/audio/speech", content=body, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 413
    assert resp.json()["detail"] == "Request body too large."


# --------------------------------------------------------------------------- #
# Voice download: atomic writes + validation + no leftovers
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, url, chunks=None, headers=None, error=None):
        self.url = url
        self._chunks = chunks if chunks is not None else []
        self.headers = headers or {}
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def iter_content(self, chunk_size=8192):
        yield from self._chunks


def _install_fake_get(monkeypatch, *, onnx_empty=False, json_invalid=False, oversized=False, http_error=False):
    """Monkeypatch requests.get to serve controlled content per URL extension."""

    def fake_get(url, stream=True, timeout=None):
        if url.endswith(".onnx.json"):
            if json_invalid:
                return _FakeResponse(url, chunks=[b"definitely-not-json"])
            return _FakeResponse(url, chunks=[b'{"sample_rate": 22050}'])
        if oversized:
            return _FakeResponse(
                url, chunks=[b"x"], headers={"content-length": str(server.MAX_VOICE_BYTES + 1)}
            )
        if http_error:
            return _FakeResponse(url, error=server.requests.HTTPError("boom"))
        if onnx_empty:
            return _FakeResponse(url, chunks=[])
        return _FakeResponse(url, chunks=[b"MODEL-BYTES"])

    monkeypatch.setattr(server.requests, "get", fake_get)


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    d = tmp_path / "voices"
    monkeypatch.setattr(server, "DATA_DIR", d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_download_voice_if_missing_success_it_atomic_and_valid(data_dir, monkeypatch):
    _install_fake_get(monkeypatch)
    onnx_path, json_path = server.get_voice_paths("en_US-test-medium")

    assert server.download_voice_if_missing("en_US-test-medium") is True
    assert onnx_path.exists() and json_path.exists()
    assert onnx_path.read_bytes() == b"MODEL-BYTES"
    assert json.loads(json_path.read_text()) == {"sample_rate": 22050}
    assert not list(data_dir.glob("*.tmp"))


def test_download_voice_if_missing_empty_model_not_published(data_dir, monkeypatch):
    _install_fake_get(monkeypatch, onnx_empty=True)
    onnx_path, json_path = server.get_voice_paths("en_US-test-medium")

    assert server.download_voice_if_missing("en_US-test-medium") is False
    assert not onnx_path.exists()
    assert not json_path.exists()
    assert not list(data_dir.glob("*.tmp"))


def test_download_voice_if_missing_http_error_not_published(data_dir, monkeypatch):
    _install_fake_get(monkeypatch, http_error=True)
    onnx_path, json_path = server.get_voice_paths("en_US-test-medium")

    assert server.download_voice_if_missing("en_US-test-medium") is False
    assert not onnx_path.exists()
    assert not json_path.exists()
    assert not list(data_dir.glob("*.tmp"))


def test_download_voice_if_missing_oversized_not_published(data_dir, monkeypatch):
    _install_fake_get(monkeypatch, oversized=True)
    onnx_path, json_path = server.get_voice_paths("en_US-test-medium")

    assert server.download_voice_if_missing("en_US-test-medium") is False
    assert not onnx_path.exists()
    assert not json_path.exists()
    assert not list(data_dir.glob("*.tmp"))


def test_download_voice_if_missing_stream_over_limit_not_published(data_dir, monkeypatch):
    # No content-length header at all; the streamed bytes themselves exceed the cap.
    monkeypatch.setattr(server, "MAX_VOICE_BYTES", 100)

    def fake_get(url, stream=True, timeout=None):
        if url.endswith(".onnx"):
            return _FakeResponse(url, chunks=[b"x" * 64] * 10)  # 640 bytes > 100
        return _FakeResponse(url, chunks=[b'{"sample_rate": 22050}'])

    monkeypatch.setattr(server.requests, "get", fake_get)
    onnx_path, json_path = server.get_voice_paths("en_US-test-medium")

    assert server.download_voice_if_missing("en_US-test-medium") is False
    assert not onnx_path.exists()
    assert not json_path.exists()
    assert not list(data_dir.glob("*.tmp"))


def test_download_voice_if_missing_invalid_json_not_published(data_dir, monkeypatch):
    _install_fake_get(monkeypatch, json_invalid=True)
    onnx_path, json_path = server.get_voice_paths("en_US-test-medium")

    assert server.download_voice_if_missing("en_US-test-medium") is False
    assert not onnx_path.exists()
    assert not json_path.exists()
    assert not list(data_dir.glob("*.tmp"))


# --------------------------------------------------------------------------- #
# Subprocess timeouts
# --------------------------------------------------------------------------- #


def test_piper_timeout_raises_500(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired("piper", server.PIPER_TIMEOUT)

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    gen = server.stream_audio_generator(["piper"], "hi", "/tmp/does-not-exist.wav", "wav")

    with pytest.raises(HTTPException) as exc:
        list(gen)
    assert exc.value.status_code == 500


def test_ffmpeg_timeout_removes_partial_mp3(monkeypatch, tmp_path):
    wav_path = tmp_path / "example.wav"
    wav_path.write_bytes(b"wav-bytes")
    mp3_path = wav_path.with_suffix(".mp3")
    mp3_path.write_bytes(b"partial")  # simulate partially-written ffmpeg output

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", server.FFMPEG_TIMEOUT)

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    with pytest.raises(HTTPException) as exc:
        server.convert_audio_if_needed(str(wav_path), "mp3")
    assert exc.value.status_code == 500
    assert not mp3_path.exists()


def test_ffmpeg_calledprocesserror_removes_partial_mp3(monkeypatch, tmp_path):
    wav_path = tmp_path / "example.wav"
    wav_path.write_bytes(b"wav-bytes")
    mp3_path = wav_path.with_suffix(".mp3")
    mp3_path.write_bytes(b"partial")

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg", output=b"", stderr=b"boom")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    with pytest.raises(HTTPException) as exc:
        server.convert_audio_if_needed(str(wav_path), "mp3")
    assert exc.value.status_code == 500
    assert not mp3_path.exists()


# --------------------------------------------------------------------------- #
# Event-loop friendliness
# --------------------------------------------------------------------------- #


def test_speech_endpoint_is_async_handler():
    """Speech handler is async (streaming-friendly) and stays a coroutine."""
    assert inspect.iscoroutinefunction(server.generate_speech) is True


def test_blocking_download_is_offloaded_to_threadpool(monkeypatch, tmp_path):
    """The voice download must be dispatched via run_in_threadpool, not run
    directly on the asyncio event loop."""
    real_run = server.run_in_threadpool
    calls = []

    def spy(func, *args, **kwargs):
        calls.append(func)
        return real_run(func, *args, **kwargs)

    monkeypatch.setattr(server, "run_in_threadpool", spy)

    data_dir = tmp_path / "voices"
    monkeypatch.setattr(server, "DATA_DIR", data_dir)

    def fake_download(voice_name):
        data_dir.mkdir(parents=True, exist_ok=True)
        return True

    monkeypatch.setattr(server, "download_voice_if_missing", fake_download)

    def fake_run(cmd, input=None, text=None, check=None, stdout=None, stderr=None, timeout=None):
        out_idx = cmd.index("--output_file") + 1
        Path(cmd[out_idx]).write_bytes(b"FAKE")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    client = TestClient(server.app)
    resp = client.post("/v1/audio/speech", json={"model": "demo-voice", "input": "hi"})
    assert resp.status_code == 200
    assert calls and calls[0] is server.download_voice_if_missing
