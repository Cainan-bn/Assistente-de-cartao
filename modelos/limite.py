"""Modelo de dados de limite de crédito."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field


class InformacaoLimite(BaseModel):
    """Representa o limite de crédito de um cartão."""

    id_cliente: str
    limite_total: Decimal = Field(ge=0)
    limite_disponivel: Decimal = Field(ge=0)
    limite_adicional: Decimal = Field(default=Decimal("0"), ge=0)
