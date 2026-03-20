"""
Middleware de rate limiting e segurança da API.

Protege os endpoints contra abuso e garante uso equitativo
dos recursos de LLM, que têm custo por token.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

_JANELA_SEGUNDOS = 60
_LIMITE_REQUISICOES_POR_JANELA = 20
_CABECALHO_CLIENTE = "x-client-id"


class MiddlewareRateLimit(BaseHTTPMiddleware):
    """
    Rate limiting por cliente com janela deslizante.

    Limita cada cliente a {_LIMITE_REQUISICOES_POR_JANELA} requisições
    por minuto, retornando 429 quando o limite é excedido.

    Em produção, o estado deve ser armazenado em Redis para
    funcionar corretamente em ambientes com múltiplas instâncias.
    """

    def __init__(self, app, requisicoes_por_janela: int = _LIMITE_REQUISICOES_POR_JANELA) -> None:
        super().__init__(app)
        self._requisicoes_por_janela = requisicoes_por_janela
        self._janelas: dict[str, list[float]] = defaultdict(list)

    def _identificar_cliente(self, request: Request) -> str:
        return (
            request.headers.get(_CABECALHO_CLIENTE)
            or request.client.host
            or "desconhecido"
        )

    def _limite_excedido(self, id_cliente: str) -> bool:
        agora = time.time()
        limite_janela = agora - _JANELA_SEGUNDOS

        self._janelas[id_cliente] = [
            ts for ts in self._janelas[id_cliente] if ts > limite_janela
        ]

        if len(self._janelas[id_cliente]) >= self._requisicoes_por_janela:
            return True

        self._janelas[id_cliente].append(agora)
        return False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in {"/health", "/docs", "/openapi.json"}:
            return await call_next(request)

        id_cliente = self._identificar_cliente(request)

        if self._limite_excedido(id_cliente):
            logger.warning("Rate limit excedido para cliente %s", id_cliente)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Limite de {self._requisicoes_por_janela} requisições por minuto excedido.",
                    "retry_after_segundos": _JANELA_SEGUNDOS,
                },
                headers={"Retry-After": str(_JANELA_SEGUNDOS)},
            )

        return await call_next(request)


class MiddlewareSeguranca(BaseHTTPMiddleware):
    """
    Adiciona cabeçalhos de segurança HTTP a todas as respostas.

    Segue as recomendações OWASP para APIs REST em ambientes bancários.
    """

    _CABECALHOS_SEGURANCA = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Cache-Control": "no-store",
        "Content-Security-Policy": "default-src 'none'",
    }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        resposta = await call_next(request)
        for cabecalho, valor in self._CABECALHOS_SEGURANCA.items():
            resposta.headers[cabecalho] = valor
        return resposta
