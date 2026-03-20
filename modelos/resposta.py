"""Modelo de resposta padronizado do assistente."""

from __future__ import annotations

from pydantic import BaseModel, Field

from modelos.estado import IntencaoUsuario


class RespostaAgente(BaseModel):
    """Resposta padronizada gerada por qualquer agente do sistema."""

    texto: str
    intencao: IntencaoUsuario
    confianca: float = Field(ge=0.0, le=1.0)
    metadados: dict = Field(default_factory=dict)
