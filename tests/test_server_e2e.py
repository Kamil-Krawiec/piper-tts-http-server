import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture
def client_with_stubs(monkeypatch, tmp_path):
    """Create a TestClient with Piper + conversion stubbed out.

    - DATA_DIR points into a temporary folder
    - download_voice_if_missing always "succeeds"
    - convert_audio_if_needed just picks media_type based on format
    - subprocess.run writes a fake WAV file instead of calling Piper
    """
    recorded_cmds = []
    recorded_formats = []
    data_dir = tmp_path / "voices"

    # Make the server use our temporary DATA_DIR
    monkeypatch.setattr(server, "DATA_DIR", data_dir)

    def fake_download_voice(_voice_name: str) -> bool:
        data_dir.mkdir(parents=True, exist_ok=True)
        return True

    def fake_convert_audio(wav_path: str, target_format: str):
        """Stub for convert_audio_if_needed.

        We only care that the server passes the correct target_format,
        so we track it and return the original WAV path.
        """
        fmt = (target_format or "wav").lower()
        recorded_formats.append(fmt)
        media_type = "audio/mpeg" if fmt == "mp3" else "audio/wav"
        return wav_path, media_type, []

    def fake_run(cmd, input=None, text=None, check=None, stdout=None, stderr=None):
        """Stub Piper CLI: write fake bytes to the output file and record the command."""
        output_index = cmd.index("--output_file") + 1
        output_path = cmd[output_index]
        Path(output_path).write_bytes(b"FAKEAUDIO")
        recorded_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(server, "download_voice_if_missing", fake_download_voice)
    monkeypatch.setattr(server, "convert_audio_if_needed", fake_convert_audio)
    monkeypatch.setattr(server.subprocess, "run", fake_run)

    client = TestClient(server.app)
    return client, recorded_cmds, recorded_formats, data_dir


def test_generate_speech_streams_audio(client_with_stubs):
    client, cmds, formats, data_dir = client_with_stubs

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "demo-voice",
            "input": ["Hello", "from Piper"],
            "format": "wav",
            "speed": 2.0,
        },
    )

    # Basic response shape
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content == b"FAKEAUDIO"

    # Piper should have been invoked once
    assert len(cmds) == 1

    # Check that length-scale is derived as 1 / speed
    assert "--length-scale" in cmds[0]
    assert cmds[0][cmds[0].index("--length-scale") + 1] == "0.5"

    # For this request we explicitly set format="wav"
    assert formats == ["wav"]

    # No temporary audio files are expected under DATA_DIR
    assert not list(data_dir.glob("piper_*.wav"))


def test_response_format_alias(client_with_stubs):
    client, _cmds, formats, data_dir = client_with_stubs

    # Only response_format is set; it should be treated as an alias for format
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "demo-voice",
            "input": "Hi there",
            "response_format": "mp3",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.content == b"FAKEAUDIO"

    # The server should have resolved the effective format to "mp3"
    assert formats == ["mp3"]

    assert not list(data_dir.glob("piper_*.wav"))


def test_missing_voice_returns_404(monkeypatch):
    # Force download_voice_if_missing to "fail"
    monkeypatch.setattr(server, "download_voice_if_missing", lambda _voice: False)

    client = TestClient(server.app)
    response = client.post(
        "/v1/audio/speech",
        json={"model": "demo-voice", "input": "hello world"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Voice 'demo-voice' not found."


def test_path_traversal_voice_name_rejected():
    client = TestClient(server.app)

    response = client.post(
        "/v1/audio/speech",
        json={"model": "../etc/passwd", "input": "malicious"},
    )

    assert response.status_code == 400
    assert "Invalid voice name" in response.json()["detail"]


def test_empty_input_rejected(client_with_stubs):
    client, _cmds, _formats, _data_dir = client_with_stubs

    response = client.post(
        "/v1/audio/speech",
        json={"model": "demo-voice", "input": ""},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "`input` text must not be empty."
