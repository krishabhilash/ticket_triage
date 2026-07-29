"""Network-free tests for the optional Gemini provider adapter."""

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from ticket_triage.constants import ALLOWED_LABELS
from ticket_triage.llm import (
    LLMClassificationError,
    LLMClassifier,
    ResponseCache,
)
from ticket_triage.llm_evaluate import main as llm_evaluate_main


@dataclass
class FakeResponse:
    text: str
    usage_metadata: object = field(
        default_factory=lambda: SimpleNamespace(
            prompt_token_count=20,
            candidates_token_count=4,
            total_token_count=24,
        )
    )


class FakeModels:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TransientError(Exception):
    status_code = 503


def make_classifier(
    outcomes: list[object], *, cache: ResponseCache | None = None
) -> tuple[LLMClassifier, FakeModels]:
    models = FakeModels(outcomes)
    client = SimpleNamespace(models=models)
    ticks = iter([10.0, 10.25, 20.0, 20.5, 30.0, 30.75])
    classifier = LLMClassifier(
        client=client,
        model="configured-by-test",
        cache=cache,
        clock=lambda: next(ticks),
        sleep=lambda _: None,
    )
    return classifier, models


def test_structured_response_is_validated_and_usage_recorded() -> None:
    classifier, models = make_classifier([FakeResponse('{"label":"fraud-report"}')])
    result = classifier.classify("I did not authorize this transfer")

    assert result.label == "fraud-report"
    assert result.latency_seconds == 0.25
    assert result.input_tokens == 20
    assert result.output_tokens == 4
    assert result.invalid_outputs == 0
    call = models.calls[0]
    assert call["model"] == "configured-by-test"
    assert "I did not authorize this transfer" in str(call["contents"])
    assert call["config"]["temperature"] == 0  # type: ignore[index]


def test_invalid_output_is_retried_without_label_fallback() -> None:
    classifier, _ = make_classifier(
        [FakeResponse('{"label":"other"}'), FakeResponse('{"label":"general"}')]
    )
    result = classifier.classify("Where can I find tax documents?")
    assert result.label == "general"
    assert result.attempts == 2
    assert result.invalid_outputs == 1


def test_transient_failure_retries_without_counting_invalid_output() -> None:
    classifier, _ = make_classifier(
        [TransientError("temporary outage"), FakeResponse('{"label":"general"}')]
    )
    result = classifier.classify("Where can I find tax documents?")
    assert result.attempts == 2
    assert result.invalid_outputs == 0


def test_bounded_invalid_outputs_raise_clear_error() -> None:
    classifier, _ = make_classifier(
        [FakeResponse("not-json"), FakeResponse("{}"), FakeResponse('{"label":"x"}')]
    )
    with pytest.raises(LLMClassificationError) as captured:
        classifier.classify("An ambiguous ticket")
    assert captured.value.attempts == 3
    assert captured.value.invalid_outputs == 3


def test_cache_avoids_second_provider_call(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.json")
    classifier, models = make_classifier(
        [FakeResponse('{"label":"account-access"}')], cache=cache
    )
    first = classifier.classify("My password is not working")
    second = classifier.classify("My password is not working")

    assert first.label == second.label == "account-access"
    assert second.cached is True
    assert len(models.calls) == 1
    assert "My password is not working" not in cache.path.read_text(encoding="utf-8")


def test_all_approved_labels_fit_structured_contract() -> None:
    assert set(ALLOWED_LABELS) == {
        "account-access",
        "transaction-dispute",
        "fraud-report",
        "general",
    }


def test_environment_configuration_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        LLMClassifier.from_env()


def test_live_evaluation_requires_explicit_authorization() -> None:
    with pytest.raises(SystemExit, match="--confirm-live"):
        llm_evaluate_main(["--data", "does-not-need-to-exist.csv"])
