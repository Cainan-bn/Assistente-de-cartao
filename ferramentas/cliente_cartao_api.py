"""
Cliente HTTP para a API REST de cartões.

Implementa o padrão Repository para isolar a lógica de acesso
à API externa, facilitando mocks em testes e troca de provedores.
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal

import httpx
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from modelos.fatura import Fatura

logger = logging.getLogger(__name__)

_TIMEOUT_SEGUNDOS = 10
_TENTATIVAS_MAXIMAS = 3


class ErroAPICartao(Exception):
    """Erro genérico ao comunicar com a API de cartões."""


class ClienteCartaoAPI:
    """
    Cliente HTTP assíncrono para a API REST de cartões.

    Implementa retry com backoff exponencial para lidar com
    instabilidades transitórias da API parceira.
    """

    def __init__(self, url_base: str = "https://api.cartoes.getronics.com") -> None:
        self._url_base = url_base
        self._cliente_http = httpx.AsyncClient(
            base_url=url_base,
            timeout=_TIMEOUT_SEGUNDOS,
        )

    @retry(
        stop=stop_after_attempt(_TENTATIVAS_MAXIMAS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def buscar_fatura_atual(self, id_cliente: str) -> Fatura:
        """
        Busca a fatura atual do cliente na API.

        Args:
            id_cliente: Identificador único do cliente.

        Returns:
            Objeto Fatura com os dados da fatura atual.

        Raises:
            ErroAPICartao: Quando a API retorna erro ou dados inválidos.
        """
        logger.info("Buscando fatura atual para cliente %s", id_cliente)
        try:
            resposta = await self._cliente_http.get(f"/clientes/{id_cliente}/fatura/atual")
            resposta.raise_for_status()
            return Fatura.model_validate(resposta.json())
        except httpx.HTTPStatusError as erro:
            raise ErroAPICartao(f"API retornou status {erro.response.status_code}") from erro
        except ValidationError as erro:
            raise ErroAPICartao("Dados de fatura inválidos recebidos da API") from erro

    @retry(
        stop=stop_after_attempt(_TENTATIVAS_MAXIMAS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def buscar_fatura_por_mes(self, id_cliente: str, mes: int, ano: int) -> Fatura:
        """
        Busca a fatura de um mês específico do cliente.

        Args:
            id_cliente: Identificador único do cliente.
            mes: Mês de referência (1-12).
            ano: Ano de referência.

        Returns:
            Objeto Fatura com os dados do período solicitado.
        """
        logger.info("Buscando fatura %d/%d para cliente %s", mes, ano, id_cliente)
        try:
            resposta = await self._cliente_http.get(
                f"/clientes/{id_cliente}/fatura",
                params={"mes": mes, "ano": ano},
            )
            resposta.raise_for_status()
            return Fatura.model_validate(resposta.json())
        except httpx.HTTPStatusError as erro:
            raise ErroAPICartao(f"API retornou status {erro.response.status_code}") from erro
        except ValidationError as erro:
            raise ErroAPICartao("Dados de fatura inválidos recebidos da API") from erro

    @retry(
        stop=stop_after_attempt(_TENTATIVAS_MAXIMAS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def buscar_limite(self, id_cliente: str):
        """Busca o limite total e disponível do cartão do cliente."""
        from modelos.limite import InformacaoLimite

        logger.info("Buscando limite para cliente %s", id_cliente)
        try:
            resposta = await self._cliente_http.get(f"/clientes/{id_cliente}/limite")
            resposta.raise_for_status()
            return InformacaoLimite.model_validate(resposta.json())
        except httpx.HTTPStatusError as erro:
            raise ErroAPICartao(f"API retornou status {erro.response.status_code}") from erro
        except ValidationError as erro:
            raise ErroAPICartao("Dados de limite inválidos recebidos da API") from erro

    @retry(
        stop=stop_after_attempt(_TENTATIVAS_MAXIMAS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def buscar_transacoes(self, id_cliente: str, quantidade: int = 10) -> list:
        """Busca as transações mais recentes do cliente."""
        from modelos.transacao import Transacao

        logger.info("Buscando %d transações para cliente %s", quantidade, id_cliente)
        try:
            resposta = await self._cliente_http.get(
                f"/clientes/{id_cliente}/transacoes",
                params={"quantidade": quantidade},
            )
            resposta.raise_for_status()
            return [Transacao.model_validate(t) for t in resposta.json()]
        except httpx.HTTPStatusError as erro:
            raise ErroAPICartao(f"API retornou status {erro.response.status_code}") from erro

    @retry(
        stop=stop_after_attempt(_TENTATIVAS_MAXIMAS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def buscar_transacoes_periodo(
        self,
        id_cliente: str,
        data_inicio: str,
        data_fim: str,
    ) -> list:
        """Busca transações de um período específico."""
        from modelos.transacao import Transacao

        logger.info(
            "Buscando transações de %s a %s para cliente %s",
            data_inicio,
            data_fim,
            id_cliente,
        )
        try:
            resposta = await self._cliente_http.get(
                f"/clientes/{id_cliente}/transacoes",
                params={"data_inicio": data_inicio, "data_fim": data_fim},
            )
            resposta.raise_for_status()
            return [Transacao.model_validate(t) for t in resposta.json()]
        except httpx.HTTPStatusError as erro:
            raise ErroAPICartao(f"API retornou status {erro.response.status_code}") from erro

    async def fechar(self) -> None:
        """Encerra a conexão HTTP de forma assíncrona."""
        await self._cliente_http.aclose()

    async def __aenter__(self) -> ClienteCartaoAPI:
        return self

    async def __aexit__(self, *args) -> None:
        await self.fechar()
