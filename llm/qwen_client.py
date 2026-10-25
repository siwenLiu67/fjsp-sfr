"""Qwen client implementing the :class:`llm.generator.LLMClient` protocol.

The client deliberately keeps authentication outside serialised state: the
DashScope key is read only from ``DASHSCOPE_API_KEY`` and is never written to
logs or cache files.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenClientError(RuntimeError):
    """Base error raised by the Qwen client."""


class QwenConfigurationError(QwenClientError):
    """Raised when the local Qwen configuration is incomplete."""


class QwenAuthenticationError(QwenClientError):
    """Raised when DashScope rejects the configured credentials."""


@dataclass(frozen=True)
class _HTTPResult:
    status_code: int
    data: dict[str, Any]
    text: str


class _RequestFailure(Exception):
    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


Transport = Callable[[str, dict[str, Any], dict[str, str], float], _HTTPResult]


def parse_enable_thinking(value: str | bool | None) -> bool | None:
    """Map CLI values to the DashScope request representation.

    ``None`` means automatic mode and therefore omits the provider-specific
    field.  Explicit true/false values are sent as JSON booleans.
    """

    if value is None or value == "auto":
        return None
    if value is True or value == "true":
        return True
    if value is False or value == "false":
        return False
    raise ValueError("enable_thinking must be one of: auto, true, false")


class QwenClient:
    """OpenAI-compatible DashScope client with retries, logging, and cache."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        enable_thinking: str | bool | None = False,
        tag: str = "default",
        log_root: str | Path = "llm_logs",
        max_attempts: int = 4,
        retry_base_seconds: float = 1.0,
        rate_limit_seconds: float = 8.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        transport: Transport | None = None,
    ) -> None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise QwenConfigurationError(
                "DASHSCOPE_API_KEY is not set; export it before using Qwen"
            )
        if not model:
            raise ValueError("model must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.enable_thinking = parse_enable_thinking(enable_thinking)
        self.base_url = os.environ.get(
            "DASHSCOPE_BASE_URL", DEFAULT_BASE_URL
        ).rstrip("/")
        self.tag = tag
        self.log_dir = Path(log_root) / tag
        self.cache_dir = self.log_dir / "cache"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        self.retry_base_seconds = float(retry_base_seconds)
        self.rate_limit_seconds = float(rate_limit_seconds)
        self._sleep = sleep_fn
        self._transport = transport
        self._api_key = api_key
        self._openai_client: Any | None = None
        self._call_index = self._discover_next_call_index()

        self.last_usage: dict[str, int | float] = {}
        self.last_elapsed_seconds = 0.0
        self.last_cache_hit = False
        self.api_request_count = 0
        self.usage_totals: dict[str, int | float] = {}

    def _discover_next_call_index(self) -> int:
        indices: list[int] = []
        for path in self.log_dir.glob("call_*_prompt.txt"):
            try:
                indices.append(int(path.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return max(indices, default=0) + 1

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "[REDACTED]")

    def _cache_key(self, prompt: str) -> str:
        identity = {
            "model": self.model,
            "temperature": self.temperature,
            "enable_thinking": self.enable_thinking,
            "prompt": prompt,
        }
        encoded = json.dumps(
            identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _write_json(self, path: Path, value: Any) -> None:
        safe_text = self._redact(
            json.dumps(value, indent=2, ensure_ascii=False, default=str)
        )
        path.write_text(safe_text + "\n", encoding="utf-8")

    def _write_call_logs(
        self,
        call_index: int,
        prompt: str,
        raw_response: dict[str, Any],
        *,
        cache_hit: bool,
    ) -> None:
        prefix = self.log_dir / f"call_{call_index:04d}"
        prefix.with_name(prefix.name + "_prompt.txt").write_text(
            self._redact(prompt), encoding="utf-8"
        )
        self._write_json(
            prefix.with_name(prefix.name + "_response.json"), raw_response
        )
        self._write_json(
            prefix.with_name(prefix.name + "_meta.json"),
            {
                "model": self.model,
                "temperature": self.temperature,
                "enable_thinking": self.enable_thinking,
                "cache_hit": cache_hit,
            },
        )

    @staticmethod
    def _usage_dict(raw_response: dict[str, Any]) -> dict[str, int | float]:
        usage = raw_response.get("usage") or {}
        if isinstance(usage, dict):
            return {
                str(key): value
                for key, value in usage.items()
                if isinstance(value, (int, float))
            }
        return {}

    @staticmethod
    def _extract_final_text(raw_response: dict[str, Any]) -> str:
        try:
            message = raw_response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise _RequestFailure(
                None, "Qwen response has no choices[0].message"
            ) from error

        content = message.get("content")
        if content is None:
            content = message.get("final_answer")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "".join(parts)
        raise _RequestFailure(None, "Qwen response has no final answer text")

    def _payload(
        self, prompt: str, enable_thinking: bool | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        return payload

    def _stdlib_transport(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _HTTPResult:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8", errors="replace")
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as error:
                    raise _RequestFailure(
                        response.status, "Qwen returned invalid JSON"
                    ) from error
                return _HTTPResult(response.status, data, text)
        except urllib.error.HTTPError as error:
            text = error.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {}
            return _HTTPResult(error.code, data, text)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise _RequestFailure(None, self._redact(str(error))) from error

    def _openai_request(
        self, prompt: str, enable_thinking: bool | None
    ) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError:
            raise

        if self._openai_client is None:
            self._openai_client = OpenAI(
                api_key=self._api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        request_arguments: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if enable_thinking is not None:
            request_arguments["extra_body"] = {
                "enable_thinking": enable_thinking
            }
        try:
            response = self._openai_client.chat.completions.create(
                **request_arguments
            )
        except Exception as error:
            status = getattr(error, "status_code", None)
            raise _RequestFailure(status, self._redact(str(error))) from error
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, dict):
            return response
        raise _RequestFailure(None, "unsupported OpenAI client response type")

    def _request_once(
        self, prompt: str, enable_thinking: bool | None
    ) -> dict[str, Any]:
        if self._transport is None:
            try:
                return self._openai_request(prompt, enable_thinking)
            except ImportError:
                transport = self._stdlib_transport
        else:
            transport = self._transport

        try:
            result = transport(
                f"{self.base_url}/chat/completions",
                self._payload(prompt, enable_thinking),
                {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                self.timeout,
            )
        except _RequestFailure:
            raise
        except Exception as error:
            raise _RequestFailure(None, self._redact(str(error))) from error
        if not 200 <= result.status_code < 300:
            message = result.text
            if result.data:
                message = json.dumps(result.data, ensure_ascii=False)
            raise _RequestFailure(
                result.status_code, self._redact(message)
            )
        return result.data

    def generate(self, prompt: str) -> str:
        """Generate one final answer, reusing a safe on-disk cache when possible."""

        call_index = self._call_index
        self._call_index += 1
        started = time.monotonic()
        prompt_path = self.log_dir / f"call_{call_index:04d}_prompt.txt"
        prompt_path.write_text(self._redact(prompt), encoding="utf-8")
        cache_path = self.cache_dir / f"{self._cache_key(prompt)}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.last_usage = dict(cached.get("usage", {}))
            self.last_elapsed_seconds = time.monotonic() - started
            self.last_cache_hit = True
            self._write_call_logs(
                call_index,
                prompt,
                cached.get("raw_response", {}),
                cache_hit=True,
            )
            return str(cached["content"])

        current_thinking = self.enable_thinking
        thinking_retry_used = False
        last_failure: _RequestFailure | None = None
        for attempt in range(self.max_attempts):
            try:
                self.api_request_count += 1
                raw_response = self._request_once(prompt, current_thinking)
                content = self._extract_final_text(raw_response)
                usage = self._usage_dict(raw_response)
                for key, value in usage.items():
                    self.usage_totals[key] = self.usage_totals.get(key, 0) + value
                cache_value = {
                    "content": content,
                    "usage": usage,
                    "raw_response": raw_response,
                }
                self._write_json(cache_path, cache_value)
                self._write_call_logs(
                    call_index, prompt, raw_response, cache_hit=False
                )
                self.last_usage = usage
                self.last_elapsed_seconds = time.monotonic() - started
                self.last_cache_hit = False
                return content
            except _RequestFailure as failure:
                last_failure = failure
                lower_message = failure.message.lower()
                if (
                    "enable_thinking" in lower_message
                    and current_thinking is not False
                    and not thinking_retry_used
                ):
                    current_thinking = False
                    thinking_retry_used = True
                    continue
                if failure.status_code == 401:
                    raise QwenAuthenticationError(
                        "Qwen authentication failed (401). Check that the key "
                        "matches DASHSCOPE_BASE_URL (China or international)."
                    ) from None
                if attempt + 1 >= self.max_attempts:
                    break
                delay = self.retry_base_seconds * (2**attempt)
                if failure.status_code == 429:
                    delay = max(self.rate_limit_seconds, delay)
                self._sleep(delay)

        assert last_failure is not None
        self._write_json(
            self.log_dir / f"call_{call_index:04d}_error.json",
            {
                "status_code": last_failure.status_code,
                "message": self._redact(last_failure.message),
                "attempts": self.max_attempts,
            },
        )
        status_text = (
            f"HTTP {last_failure.status_code}: "
            if last_failure.status_code is not None
            else ""
        )
        raise QwenClientError(
            f"Qwen request failed after {self.max_attempts} attempts: "
            f"{status_text}{self._redact(last_failure.message)}"
        ) from None


__all__ = [
    "DEFAULT_BASE_URL",
    "QwenAuthenticationError",
    "QwenClient",
    "QwenClientError",
    "QwenConfigurationError",
    "parse_enable_thinking",
]
