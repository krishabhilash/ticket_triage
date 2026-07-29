"""Optional, provider-isolated Gemini ticket classifier.

The Google SDK is imported only by ``from_env``. Classical training and
inference therefore require neither this optional dependency nor network access.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any

from ticket_triage.constants import ALLOWED_LABELS
from ticket_triage.predictor import validate_message

_PROMPT_VERSION = "ticket-routing-v1"
_LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": list(ALLOWED_LABELS)}},
    "required": ["label"],
    "additionalProperties": False,
}
_PROMPT = """Classify this support ticket into exactly one route.

Labels:
- account-access: problems signing in, passwords, verification codes, or locked accounts
- transaction-dispute: a recognized transaction is delayed, incorrect, duplicated, unexpectedly priced, or missing
- fraud-report: unauthorized activity, account compromise, phishing, scams, theft, or impersonation
- general: informational requests not covered by the other routes

Important distinction: A recognized or user-initiated transaction with a processing problem is transaction-dispute. Unauthorized, deceptive, or account-compromise activity is fraud-report.

Ticket:
{ticket}"""


class LLMClassificationError(RuntimeError):
    """Raised after bounded attempts fail without an approved label."""

    def __init__(self, message: str, *, attempts: int, invalid_outputs: int) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.invalid_outputs = invalid_outputs


@dataclass(frozen=True)
class LLMResult:
    """One Gemini classification and available operational metadata."""

    label: str
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    attempts: int
    invalid_outputs: int
    cached: bool = False


class ResponseCache:
    """Small JSON cache keyed by hashes of model, prompt version, and ticket."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid LLM cache file: {self.path}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Invalid LLM cache file: {self.path}")
        return value

    def get(self, key: str) -> LLMResult | None:
        """Return a validated cached result when present."""
        payload = self._read().get(key)
        if payload is None:
            return None
        try:
            result = LLMResult(**payload, cached=True)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid cached LLM response") from error
        if result.label not in ALLOWED_LABELS:
            raise ValueError("Cached LLM response contains an unknown label")
        return result

    def put(self, key: str, result: LLMResult) -> None:
        """Persist response metadata without storing ticket text or secrets."""
        entries = self._read()
        payload = asdict(result)
        payload.pop("cached")
        entries[key] = payload
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


class LLMClassifier:
    """Provider-isolated Gemini classifier with strict structured validation."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_attempts: int = 3,
        cache: ResponseCache | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model.strip():
            raise ValueError("Gemini model name must not be empty")
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        self._client = client
        self.model = model
        self.max_attempts = max_attempts
        self._cache = cache
        self._clock = clock
        self._sleep = sleep

    @classmethod
    def from_env(
        cls, *, max_attempts: int = 3, cache_path: str | Path | None = None
    ) -> "LLMClassifier":
        """Build a Gemini classifier from environment configuration."""
        api_key = os.environ.get("GEMINI_API_KEY")
        model = os.environ.get("GEMINI_MODEL")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        if not model:
            raise RuntimeError("GEMINI_MODEL is not configured")
        try:
            from google import genai
        except ImportError as error:
            raise RuntimeError(
                "Gemini support is optional; install the project with [llm]"
            ) from error
        cache = ResponseCache(cache_path) if cache_path is not None else None
        try:
            client = genai.Client(api_key=api_key)
        except Exception:
            raise RuntimeError("Unable to initialize the Gemini client") from None
        return cls(
            client=client,
            model=model,
            max_attempts=max_attempts,
            cache=cache,
        )

    def classify(self, text: object) -> LLMResult:
        """Classify one ticket, retrying only invalid or transient failures."""
        message = validate_message(text)
        cache_key = sha256(
            f"{_PROMPT_VERSION}\0{self.model}\0{message}".encode("utf-8")
        ).hexdigest()
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        invalid_outputs = 0
        for attempt in range(1, self.max_attempts + 1):
            started = self._clock()
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=_PROMPT.format(ticket=message),
                    config={
                        "temperature": 0,
                        "response_mime_type": "application/json",
                        "response_json_schema": _LABEL_SCHEMA,
                    },
                )
                result = self._parse_response(
                    response,
                    latency_seconds=max(0.0, self._clock() - started),
                    attempts=attempt,
                    invalid_outputs=invalid_outputs,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                invalid_outputs += 1
                if attempt == self.max_attempts:
                    break
                self._sleep(min(2 ** (attempt - 1), 4))
                continue
            except Exception as error:
                if not _is_transient(error) or attempt == self.max_attempts:
                    raise LLMClassificationError(
                        "Gemini classification failed",
                        attempts=attempt,
                        invalid_outputs=invalid_outputs,
                    ) from None
                self._sleep(min(2 ** (attempt - 1), 4))
                continue

            if self._cache is not None:
                self._cache.put(cache_key, result)
            return result

        raise LLMClassificationError(
            "Gemini returned invalid output after bounded retries",
            attempts=self.max_attempts,
            invalid_outputs=invalid_outputs,
        ) from None

    @staticmethod
    def _parse_response(
        response: Any,
        *,
        latency_seconds: float,
        attempts: int,
        invalid_outputs: int,
    ) -> LLMResult:
        payload = json.loads(response.text)
        if not isinstance(payload, dict) or set(payload) != {"label"}:
            raise ValueError("Gemini response must contain only label")
        label = payload["label"]
        if label not in ALLOWED_LABELS:
            raise ValueError("Gemini response contains an unknown label")
        usage = getattr(response, "usage_metadata", None)
        return LLMResult(
            label=label,
            latency_seconds=latency_seconds,
            input_tokens=_optional_int(usage, "prompt_token_count"),
            output_tokens=_optional_int(usage, "candidates_token_count"),
            total_tokens=_optional_int(usage, "total_token_count"),
            attempts=attempts,
            invalid_outputs=invalid_outputs,
        )


def _optional_int(value: Any, attribute: str) -> int | None:
    item = getattr(value, attribute, None)
    return int(item) if item is not None else None


def _is_transient(error: Exception) -> bool:
    status = getattr(error, "status_code", getattr(error, "code", None))
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    return type(error).__name__ in {
        "ServerError",
        "ServiceUnavailable",
        "TooManyRequests",
    }
