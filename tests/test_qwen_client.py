import json
from types import SimpleNamespace

import pytest

from fjsp_sfr.llm.qwen_client import (
    QwenAuthenticationError,
    QwenClient,
    QwenClientError,
    QwenConfigurationError,
)


def _response(content="pong", *, usage=None, reasoning="hidden reasoning"):
    data = {
        "choices": [
            {
                "message": {
                    "content": content,
                    "reasoning_content": reasoning,
                }
            }
        ],
        "usage": usage or {"prompt_tokens": 2, "completion_tokens": 1},
    }
    return SimpleNamespace(status_code=200, data=data, text=json.dumps(data))


def _client(tmp_path, monkeypatch, transport, **kwargs):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-test-key")
    return QwenClient(
        model="qwen-test",
        temperature=0.8,
        max_tokens=128,
        timeout=3,
        tag="unit",
        log_root=tmp_path,
        transport=transport,
        retry_base_seconds=1,
        rate_limit_seconds=8,
        **kwargs,
    )


def test_missing_key_has_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(QwenConfigurationError, match="DASHSCOPE_API_KEY"):
        QwenClient(
            model="qwen-test",
            temperature=0.8,
            max_tokens=32,
            timeout=2,
            log_root=tmp_path,
        )


def test_retry_and_rate_limit_wait(tmp_path, monkeypatch):
    responses = [
        SimpleNamespace(status_code=500, data={}, text="temporary"),
        SimpleNamespace(status_code=429, data={}, text="rate limited"),
        _response("ok"),
    ]
    sleeps = []

    def transport(*_args):
        return responses.pop(0)

    client = _client(
        tmp_path,
        monkeypatch,
        transport,
        sleep_fn=sleeps.append,
    )
    assert client.generate("hello") == "ok"
    assert sleeps == [1, 8]


def test_cache_hit_avoids_second_request(tmp_path, monkeypatch):
    calls = []

    def transport(*args):
        calls.append(args)
        return _response("cached answer")

    client = _client(tmp_path, monkeypatch, transport)
    assert client.generate("same prompt") == "cached answer"
    assert not client.last_cache_hit
    assert client.generate("same prompt") == "cached answer"
    assert client.last_cache_hit
    assert len(calls) == 1
    assert len(list((tmp_path / "unit").glob("call_*_prompt.txt"))) == 2


def test_enable_thinking_error_retries_explicit_false(tmp_path, monkeypatch):
    payloads = []

    def transport(_url, payload, _headers, _timeout):
        payloads.append(payload)
        if len(payloads) == 1:
            return SimpleNamespace(
                status_code=400,
                data={"error": {"message": "enable_thinking is required"}},
                text="enable_thinking is required",
            )
        return _response("final only", reasoning="do not return me")

    client = _client(
        tmp_path,
        monkeypatch,
        transport,
        enable_thinking="auto",
    )
    assert client.generate("think") == "final only"
    assert "enable_thinking" not in payloads[0]
    assert payloads[1]["enable_thinking"] is False


def test_key_never_written_to_logs_or_cache(tmp_path, monkeypatch):
    key = "secret-test-key"

    def transport(*_args):
        return _response("safe result")

    client = _client(tmp_path, monkeypatch, transport)
    client.generate("ordinary prompt")

    for path in (tmp_path / "unit").rglob("*"):
        if path.is_file():
            assert key not in path.read_text(encoding="utf-8")


def test_key_is_redacted_from_errors(tmp_path, monkeypatch):
    key = "secret-test-key"

    def transport(*_args):
        return SimpleNamespace(
            status_code=500,
            data={"error": f"provider echoed {key}"},
            text=f"provider echoed {key}",
        )

    client = _client(
        tmp_path,
        monkeypatch,
        transport,
        max_attempts=1,
    )
    with pytest.raises(QwenClientError) as caught:
        client.generate("fail safely")
    assert key not in str(caught.value)
    for path in (tmp_path / "unit").rglob("*"):
        if path.is_file():
            assert key not in path.read_text(encoding="utf-8")


def test_401_explains_key_and_region_mismatch(tmp_path, monkeypatch):
    def transport(*_args):
        return SimpleNamespace(status_code=401, data={}, text="unauthorized")

    client = _client(tmp_path, monkeypatch, transport)
    with pytest.raises(QwenAuthenticationError, match="China or international"):
        client.generate("authenticate")
