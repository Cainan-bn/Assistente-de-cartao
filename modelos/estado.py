"""Modelos de estado da conversa e enumerações de intenção."""

from __future__ import annotations

from enum import StrEnum

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class IntencaoUsuario(StrEnum):
    """Enumeração das intenções suportadas pelo assistente."""

    CONSULTA_FATURA = "CONSULTA_FATURA"
    CONSULTA_LIMITE = "CONSULTA_LIMITE"
    CONSULTA_TRANSACOES = "CONSULTA_TRANSACOES"
    GENERICA = "GENERICA"


class EstadoConversa(BaseModel):
    """Estado completo de uma sessão de conversa com o assistente."""

    id_sessao: str
    id_cliente: str
    historico_mensagens: list[BaseMessage] = Field(default_factory=list)
    intencao_atual: IntencaoUsuario | None = None
