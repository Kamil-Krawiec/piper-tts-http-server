import pytest
from fastapi import HTTPException

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
