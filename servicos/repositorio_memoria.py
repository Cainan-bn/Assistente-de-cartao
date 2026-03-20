"""
Repositório de memória de conversa com Redis.

Persiste o histórico de mensagens entre requisições, permitindo
que o assistente mantenha contexto ao longo de múltiplos turnos.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

import redis.asyncio as redis
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

_TTL_SESSAO_SEGUNDOS = 3600  # 1 hora


class RepositorioMemoria(ABC):
    """Interface abstrata para repositórios de memória de conversa."""

    @abstractmethod
    async def carregar_historico(self, id_sessao: str) -> list[BaseMessage]:
        """Carrega o histórico de mensagens de uma sessão."""

    @abstractmethod
    async def salvar_mensagem(self, id_sessao: str, mensagem: BaseMessage) -> None:
        """Persiste uma nova mensagem no histórico da sessão."""

    @abstractmethod
    async def limpar_sessao(self, id_sessao: str) -> None:
        """Remove todas as mensagens de uma sessão."""


class RepositorioMemoriaRedis(RepositorioMemoria):
    """
    Implementação do repositório de memória usando Redis.

    Serializa mensagens como JSON e usa TTL para expirar sessões inativas,
    evitando acúmulo indefinido de dados.
    """

    def __init__(self, cliente_redis: redis.Redis) -> None:
        self._redis = cliente_redis

    def _chave_sessao(self, id_sessao: str) -> str:
        return f"sessao:{id_sessao}:historico"

    def _serializar_mensagem(self, mensagem: BaseMessage) -> str:
        return json.dumps({
            "tipo": mensagem.__class__.__name__,
            "conteudo": mensagem.content,
        })

    def _deserializar_mensagem(self, dado: str) -> BaseMessage:
        obj = json.loads(dado)
        mapeamento = {
            "HumanMessage": HumanMessage,
            "AIMessage": AIMessage,
        }
        classe = mapeamento.get(obj["tipo"], HumanMessage)
        return classe(content=obj["conteudo"])

    async def carregar_historico(self, id_sessao: str) -> list[BaseMessage]:
        """
        Carrega o histórico de mensagens da sessão no Redis.

        Args:
            id_sessao: Identificador único da sessão de conversa.

        Returns:
            Lista de mensagens em ordem cronológica.
        """
        try:
            chave = self._chave_sessao(id_sessao)
            dados = await self._redis.lrange(chave, 0, -1)
            return [self._deserializar_mensagem(d) for d in dados]
        except Exception:
            logger.exception("Falha ao carregar histórico da sessão %s", id_sessao)
            return []

    async def salvar_mensagem(self, id_sessao: str, mensagem: BaseMessage) -> None:
        """
        Persiste uma mensagem no histórico e renova o TTL da sessão.

        Args:
            id_sessao: Identificador único da sessão de conversa.
            mensagem: Mensagem a ser persistida.
        """
        try:
            chave = self._chave_sessao(id_sessao)
            serializada = self._serializar_mensagem(mensagem)
            pipe = self._redis.pipeline()
            pipe.rpush(chave, serializada)
            pipe.expire(chave, _TTL_SESSAO_SEGUNDOS)
            await pipe.execute()
        except Exception:
            logger.exception("Falha ao salvar mensagem na sessão %s", id_sessao)

    async def limpar_sessao(self, id_sessao: str) -> None:
        """Remove o histórico completo de uma sessão."""
        try:
            await self._redis.delete(self._chave_sessao(id_sessao))
            logger.info("Sessão %s limpa com sucesso", id_sessao)
        except Exception:
            logger.exception("Falha ao limpar sessão %s", id_sessao)


class RepositorioMemoriaEmMemoria(RepositorioMemoria):
    """
    Implementação em memória para uso em testes e desenvolvimento local.

    Não persiste entre reinicializações — útil para evitar dependência
    do Redis em ambientes de CI.
    """

    def __init__(self) -> None:
        self._historicos: dict[str, list[BaseMessage]] = {}

    async def carregar_historico(self, id_sessao: str) -> list[BaseMessage]:
        return list(self._historicos.get(id_sessao, []))

    async def salvar_mensagem(self, id_sessao: str, mensagem: BaseMessage) -> None:
        if id_sessao not in self._historicos:
            self._historicos[id_sessao] = []
        self._historicos[id_sessao].append(mensagem)

    async def limpar_sessao(self, id_sessao: str) -> None:
        self._historicos.pop(id_sessao, None)
