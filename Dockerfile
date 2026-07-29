FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/artifacts/model.joblib

WORKDIR /app

COPY requirements/runtime.txt /app/requirements/runtime.txt
RUN python -m pip install --no-cache-dir --requirement requirements/runtime.txt

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --create-home --shell /usr/sbin/nologin appuser \
    && mkdir --parents /app/artifacts \
    && chown --recursive appuser:appuser /app

COPY --chown=appuser:appuser src /app/src

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).read()"]

CMD ["uvicorn", "ticket_triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
