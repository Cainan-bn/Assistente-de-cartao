"""
Classificador de intenção do usuário via prompt engineering estruturado.

Utiliza few-shot prompting e structured outputs para garantir
classificações consistentes e auditáveis.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from modelos.estado import IntencaoUsuario

logger = logging.getLogger(__name__)

_EXEMPLOS_CLASSIFICACAO = """
Exemplos de classificação:

Mensagem: "qual o valor da minha fatura de dezembro?"
Intenção: CONSULTA_FATURA

Mensagem: "quanto tenho de limite disponível?"
Intenção: CONSULTA_LIMITE

Mensagem: "mostra meus gastos dos últimos 30 dias"
Intenção: CONSULTA_TRANSACOES

Mensagem: "olá, tudo bem?"
Intenção: GENERICA
"""

_PROMPT_CLASSIFICACAO = ChatPromptTemplate.from_messages([
    (
        "system",
        """Você é um classificador de intenções para um assistente de cartões de crédito.
        Sua única função é identificar a intenção do usuário dentre as categorias disponíveis.
        Seja preciso e consistente.

        Categorias disponíveis:
        - CONSULTA_FATURA: perguntas sobre faturas, valores devidos, vencimento
        - CONSULTA_LIMITE: perguntas sobre limite de crédito, disponível ou total
        - CONSULTA_TRANSACOES: perguntas sobre compras, lançamentos, gastos
        - GENERICA: qualquer outra mensagem

        {exemplos}
        """,
    ),
    ("human", "Classifique a intenção desta mensagem: {mensagem}"),
])


class _SaidaClassificacao(BaseModel):
    """Schema de saída estruturada para o classificador."""

    intencao: IntencaoUsuario = Field(description="A intenção identificada na mensagem do usuário")
    confianca: float = Field(ge=0.0, le=1.0, description="Nível de confiança entre 0 e 1")


class ClassificadorIntencao:
    """
    Classifica a intenção do usuário usando LLM com structured output.

    Usa few-shot prompting para garantir consistência nas classificações
    e retorna um score de confiança para monitoramento.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        llm_estruturado = llm.with_structured_output(_SaidaClassificacao)
        self._cadeia = _PROMPT_CLASSIFICACAO | llm_estruturado

    async def classificar(self, mensagem: str) -> IntencaoUsuario:
        """
        Classifica a intenção de uma mensagem do usuário.

        Args:
            mensagem: Texto enviado pelo usuário.

        Returns:
            A IntencaoUsuario identificada.
        """
        try:
            resultado: _SaidaClassificacao = await self._cadeia.ainvoke({
                "mensagem": mensagem,
                "exemplos": _EXEMPLOS_CLASSIFICACAO,
            })
            logger.debug(
                "Classificação: %s (confiança: %.2f)",
                resultado.intencao,
                resultado.confianca,
            )
            return resultado.intencao
        except Exception:
            logger.exception("Falha ao classificar intenção. Retornando GENERICA.")
            return IntencaoUsuario.GENERICA
