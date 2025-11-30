import pytest
from fastapi import HTTPException
from pathlib import Path

import server


def test_get_voice_paths_respects_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)

    onnx_path, json_path = server.get_voice_paths("en_US-test-low")

    assert onnx_path == tmp_path / "en_US-test-low.onnx"
    assert json_path == tmp_path / "en_US-test-low.onnx.json"


def test_get_voice_paths_rejects_path_traversal():
    with pytest.raises(ValueError):
        server.get_voice_paths("../etc/passwd")

    with pytest.raises(ValueError):
        server.get_voice_paths("..\\windows")

    with pytest.raises(ValueError):
        server.get_voice_paths(".hidden")


def test_convert_audio_if_needed_rejects_unknown_format():
    # Unsupported target format should raise 400 and never call ffmpeg
    with pytest.raises(HTTPException) as excinfo:
        server.convert_audio_if_needed("/tmp/example.wav", "flac")

    assert excinfo.value.status_code == 400


def test_convert_audio_if_needed_converts_to_mp3(monkeypatch, tmp_path):
    wav_path = tmp_path / "example.wav"
    wav_path.write_bytes(b"wav-bytes")

    recorded_cmd = {}

    def fake_run(cmd, check=None, stdout=None, stderr=None):
        recorded_cmd["cmd"] = cmd
        Path(str(wav_path).replace(".wav", ".mp3")).write_bytes(b"mp3-bytes")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    final_path, media_type, cleanup = server.convert_audio_if_needed(
        str(wav_path), "mp3"
    )

    expected_mp3 = str(wav_path).replace(".wav", ".mp3")
    assert final_path == expected_mp3
    assert media_type == "audio/mpeg"
    assert cleanup == [expected_mp3]
    assert Path(expected_mp3).read_bytes() == b"mp3-bytes"

    cmd = recorded_cmd["cmd"]
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-i") + 1] == str(wav_path)


def test_download_voice_if_missing_uses_existing_files(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)

    voice_name = "en_US-test-low"
    onnx_path, json_path = server.get_voice_paths(voice_name)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(b"onnx-bytes")
    json_path.write_text("{}")

    # When both files exist locally, no network calls should be needed
    assert server.download_voice_if_missing(voice_name) is True


def test_download_voice_if_missing_rejects_bad_name(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)

    # Voice name that does not match the expected pattern should be rejected
    assert server.download_voice_if_missing("badvoice") is False
