"""Modelo de dados de transação do cartão."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class Transacao(BaseModel):
    """Representa um lançamento no cartão de crédito."""

    id_transacao: str
    id_cliente: str
    descricao: str
    valor: Decimal = Field(ge=0)
    data_hora: datetime
    categoria: str
    parcela_atual: int | None = None
    total_parcelas: int | None = None
    status: str  # "APROVADA" | "PENDENTE" | "CANCELADA"
