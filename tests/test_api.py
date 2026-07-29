"""HTTP behavior tests for the thin FastAPI transport."""

import io
import logging
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ticket_triage.api import MAX_BATCH_SIZE, MAX_MESSAGE_LENGTH, create_app
from ticket_triage.constants import ALLOWED_LABELS
from ticket_triage.model import build_pipeline, fit_pipeline, save_pipeline
from ticket_triage.observability import JsonFormatter


@pytest.fixture()
def model_path(tmp_path: Path) -> Path:
    """Create a tiny deterministic four-class model."""
    frame = pd.DataFrame(
        {
            "text": [
                "I cannot log in",
                "My password is blocked",
                "My recognized transfer is delayed",
                "I was charged twice",
                "Someone stole my funds",
                "I never authorized this withdrawal",
                "How does staking work",
                "What assets are supported",
            ],
            "label": [
                "account-access",
                "account-access",
                "transaction-dispute",
                "transaction-dispute",
                "fraud-report",
                "fraud-report",
                "general",
                "general",
            ],
        }
    )
    pipeline = fit_pipeline(build_pipeline(class_weight="balanced"), frame)
    return save_pipeline(pipeline, tmp_path / "model.joblib")


@pytest.fixture()
def client(model_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Start the API with a valid temporary model."""
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_ready_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_predict_returns_approved_label_and_confidence(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "I cannot log in"})
    assert response.status_code == 200
    assert response.json()["label"] in ALLOWED_LABELS
    assert 0.0 <= response.json()["confidence"] <= 1.0
    assert isinstance(response.json()["requires_review"], bool)


@pytest.mark.parametrize("text", ["", "   "])
def test_predict_rejects_empty_text(client: TestClient, text: str) -> None:
    assert client.post("/predict", json={"text": text}).status_code == 422


def test_predict_rejects_missing_text(client: TestClient) -> None:
    assert client.post("/predict", json={}).status_code == 422


def test_predict_rejects_non_string(client: TestClient) -> None:
    assert client.post("/predict", json={"text": 123}).status_code == 422


def test_predict_rejects_oversized_message(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "x" * (MAX_MESSAGE_LENGTH + 1)})
    assert response.status_code == 422


def test_batch_prediction_preserves_order_and_shape(client: TestClient) -> None:
    response = client.post(
        "/predict/batch",
        json={"texts": ["I cannot log in", "I never authorized this withdrawal"]},
    )
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert all(item["label"] in ALLOWED_LABELS for item in predictions)
    assert all(0.0 <= item["confidence"] <= 1.0 for item in predictions)
    assert all(isinstance(item["requires_review"], bool) for item in predictions)


def test_low_confidence_requires_review(
    model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("LOW_CONFIDENCE_THRESHOLD", "1")
    with TestClient(create_app()) as review_client:
        result = review_client.post("/predict", json={"text": "I cannot log in"})
    assert result.json()["requires_review"] is True


def test_high_confidence_does_not_require_review(
    model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("LOW_CONFIDENCE_THRESHOLD", "0")
    with TestClient(create_app()) as review_client:
        result = review_client.post("/predict", json={"text": "I cannot log in"})
    assert result.json()["requires_review"] is False


def test_request_logging_excludes_ticket_text(client: TestClient) -> None:
    private_text = "PRIVATE-TICKET-CONTENT-12345"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    api_logger = logging.getLogger("ticket_triage.api")
    api_logger.addHandler(handler)
    try:
        response = client.post(
            "/predict",
            headers={"X-Request-ID": "safe-request-123"},
            json={"text": private_text},
        )
    finally:
        api_logger.removeHandler(handler)
    captured = stream.getvalue()
    assert response.headers["X-Request-ID"] == "safe-request-123"
    assert private_text not in captured
    assert "request_completed" in captured
    assert "safe-request-123" in captured


def test_batch_rejects_empty_list(client: TestClient) -> None:
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422


def test_batch_rejects_oversized_list(client: TestClient) -> None:
    payload = {"texts": ["valid message"] * (MAX_BATCH_SIZE + 1)}
    assert client.post("/predict/batch", json=payload).status_code == 422


def test_model_not_ready_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(create_app()) as unavailable_client:
        health = unavailable_client.get("/health")
        ready = unavailable_client.get("/ready")
        prediction = unavailable_client.post("/predict", json={"text": "Help"})

    assert health.status_code == 200
    assert health.json() == {"status": "healthy", "model_loaded": False}
    assert ready.status_code == 503
    assert ready.json() == {"detail": "Model is not ready"}
    assert prediction.status_code == 503
    assert prediction.json() == {"detail": "Model is not ready"}
