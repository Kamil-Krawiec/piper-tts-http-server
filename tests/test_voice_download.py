from pathlib import Path

import pytest
from requests import Response

import server


VOICE = "en_US-some-voice-medium"


@pytest.fixture
def remote_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    assets = {".onnx": b"model", ".onnx.json": b"{}"}
    urls = []

    def get(url, *, stream, timeout):
        urls.append(url)
        suffix = ".onnx.json" if url.endswith(".json") else ".onnx"
        response = Response()
        response.status_code = 200
        response._content = assets[suffix]
        response._content_consumed = True
        return response

    monkeypatch.setattr(server.requests, "get", get)
    return assets, urls


def test_download_publishes_complete_pair_and_reuses_cache(remote_voice, tmp_path):
    _, urls = remote_voice
    base = f"{server.HF_URL_BASE}/en/en_US/some-voice/medium/{VOICE}"
    assert server.download_voice_if_missing(VOICE)
    assert server.download_voice_if_missing(VOICE)
    assert urls == [base + ".onnx", base + ".onnx.json"]
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == {
        f"{VOICE}.onnx": b"model",
        f"{VOICE}.onnx.json": b"{}",
    }


@pytest.mark.parametrize("metadata", [b"", b"invalid", b"[]", b" " * 9 + b"{}"])
def test_invalid_download_leaves_no_files(monkeypatch, remote_voice, tmp_path, metadata):
    assets, _ = remote_voice
    assets[".onnx.json"] = metadata
    monkeypatch.setattr(server, "METADATA_BYTES_LIMIT", 8)
    assert not server.download_voice_if_missing(VOICE)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing_suffix", [None, ".onnx", ".onnx.json"])
def test_second_publish_failure_rolls_back_new_files(
    monkeypatch, remote_voice, tmp_path, existing_suffix
):
    if existing_suffix:
        (tmp_path / (VOICE + existing_suffix)).write_bytes(b"original")

    replace = Path.replace
    replacements = []

    def replace_second_fails(path, target):
        replacements.append(target)
        if len(replacements) == 2:
            raise OSError("simulated publication failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", replace_second_fails)

    assert not server.download_voice_if_missing(VOICE)
    assert len(replacements) == 2
    expected = {VOICE + existing_suffix: b"original"} if existing_suffix else {}
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == expected
