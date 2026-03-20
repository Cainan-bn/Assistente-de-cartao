"""Modelos de dados do domínio de cartões."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field


class Fatura(BaseModel):
    """Representa uma fatura do cartão de crédito."""

    id_fatura: str
    id_cliente: str
    valor_total: Decimal = Field(ge=0)
    valor_minimo: Decimal = Field(ge=0)
    data_vencimento: date
    data_fechamento: date
    status: str  # "ABERTA" | "FECHADA" | "PAGA" | "VENCIDA"
    mes_referencia: int = Field(ge=1, le=12)
    ano_referencia: int
