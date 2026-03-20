"""
Configuração de logging estruturado e middleware de observabilidade.

Emite logs em formato JSON para integração com stacks de observabilidade
como Datadog, ELK ou Cloud Logging (GCP/AWS).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

_CAMPOS_SENSIVEIS = frozenset({"authorization", "x-api-key", "cookie"})


def configurar_logging(nivel: str = "INFO") -> None:
    """
    Configura logging estruturado em JSON com structlog.

    Deve ser chamado uma única vez na inicialização da aplicação.
    Os logs são emitidos em JSON para facilitar ingestão por ferramentas
    de observabilidade como Datadog, Splunk ou Cloud Logging.

    Args:
        nivel: Nível de log (DEBUG, INFO, WARNING, ERROR).
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, nivel.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


class MiddlewareObservabilidade(BaseHTTPMiddleware):
    """
    Middleware que adiciona logging estruturado e correlation ID a cada requisição.

    Emite um log de entrada e um de saída com duração, status e metadados
    da requisição, vinculados por um correlation ID rastreável.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = request.headers.get("x-correlation-id", str(uuid.uuid4()))
        inicio = time.perf_counter()

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            metodo=request.method,
            caminho=request.url.path,
        )

        log = structlog.get_logger()
        log.info("requisicao_recebida")

        try:
            resposta = await call_next(request)
        except Exception:
            log.exception("erro_nao_tratado")
            raise

        duracao_ms = round((time.perf_counter() - inicio) * 1000, 2)
        log.info(
            "requisicao_concluida",
            status=resposta.status_code,
            duracao_ms=duracao_ms,
        )

        resposta.headers["X-Correlation-ID"] = correlation_id
        return resposta
