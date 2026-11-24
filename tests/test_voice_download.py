from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

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


@pytest.mark.parametrize(
    "voice, remote_dir",
    [
        (VOICE, "en/en_US/some-voice/medium"),
        ("pt_PT-tugão-medium", "pt/pt_PT/tugão/medium"),
    ],
)
def test_download_publishes_complete_pair_and_reuses_cache(
    remote_voice, tmp_path, voice, remote_dir
):
    _, urls = remote_voice
    base = f"{server.HF_URL_BASE}/{remote_dir}/{voice}"
    assert server.download_voice_if_missing(voice)
    assert server.download_voice_if_missing(voice)
    assert urls == [base + ".onnx", base + ".onnx.json"]
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == {
        f"{voice}.onnx": b"model",
        f"{voice}.onnx.json": b"{}",
    }


def test_cached_voice_is_available_during_another_download(monkeypatch, remote_voice):
    cached_voice = "custom_voice"
    onnx_path, json_path = server.get_voice_paths(cached_voice)
    onnx_path.write_bytes(b"model")
    json_path.write_text("{}")

    download_started = Event()
    release_download = Event()
    get = server.requests.get

    def paused_get(*args, **kwargs):
        download_started.set()
        assert release_download.wait(timeout=10)
        return get(*args, **kwargs)

    monkeypatch.setattr(server.requests, "get", paused_get)
    with ThreadPoolExecutor(max_workers=2) as executor:
        downloading = executor.submit(server.download_voice_if_missing, VOICE)
        try:
            assert download_started.wait(timeout=5)
            cached = executor.submit(server.download_voice_if_missing, cached_voice)
            assert cached.result(timeout=5)
        finally:
            release_download.set()
        assert downloading.result(timeout=5)


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
