FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

FROM base AS dependencias
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

FROM dependencias AS producao
COPY . .
RUN useradd --no-create-home --shell /bin/false agente
USER agente

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "servicos.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
